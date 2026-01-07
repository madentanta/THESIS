<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Http;
use Carbon\Carbon;
use Exception;

class ResultController extends Controller
{
    /**
     * Ambil semua rekomendasi tanaman berdasarkan input_id
     * Terintegrasi dengan FastAPI (Hopular) di Docker VPS
     */
    public function getRecommendation($input_id)
    {
        // 1. Hardcode care instructions untuk Response FE
        $careManual = [
            "Tebu" => "1. Siram tiap 3 hari.\n2. Pupuk NPK tiap 2 minggu.\n3. Pastikan tanah gembur dan subur.\n4. Jaga jarak tanam 1.2 m antar tanaman.\n5. Panen setelah 10-12 bulan.",
            "Jagung" => "1. Siram tiap 2 hari, terutama saat musim kemarau.\n2. Pupuk urea atau NPK tiap 1 minggu.\n3. Tanam di tanah yang kaya organik.\n4. Pastikan mendapat sinar matahari penuh.\n5. Jaga jarak tanam 25-30 cm per tanaman.\n6. Panen setelah 3-4 bulan.",
            "Padi" => "1. Sirami rutin, jaga air tetap tergenang 5-10 cm.\n2. Pupuk NPK dan pupuk organik sesuai dosis.\n3. Kontrol hama seperti wereng dan penggerek batang.\n4. Pastikan padi mendapat sinar matahari cukup.\n5. Panen setelah 4-5 bulan."
        ];

        // 2. Cek apakah rekomendasi sudah ada di DB untuk menghindari double hit ke AI
        $existing = DB::table('crop_recommendation')
            ->where('input_id', $input_id)
            ->get();

        if ($existing->isNotEmpty()) {
            $recommendations = $existing->map(function($item) use ($careManual) {
                return [
                    "nama_tanaman" => $item->recommended_crop,
                    "care_instructions" => $careManual[$item->recommended_crop] ?? ($item->care_instructions ?? "Instruksi tidak tersedia.")
                ];
            });

            return response()->json([
                "success" => true,
                "input_id" => $input_id,
                "recommendations" => $recommendations
            ]);
        }

        // 3. Ambil data input dari database
        $inputData = DB::table('input')->where('input_id', $input_id)->first();

        if (!$inputData) {
            return response()->json([
                "success" => false,
                "message" => "Input data tidak ditemukan."
            ], 404);
        }

        try {
            // 4. Panggil Hopular API (FastAPI) via Internal Docker Network
            // URL menggunakan nama service 'plantadvisor_ai' port 8001
            $aiResponse = Http::timeout(60) 
                ->retry(2, 1000)
                ->post("http://ai:8001/inference", [
                    "input_data" => [
                        [
                            "soil_ph" => floatval($inputData->soil_ph),
                            "temperature" => floatval($inputData->temperature),
                            "humidity" => floatval($inputData->humidity),
                            "location" => $inputData->location,
                            "previous_crop" => $inputData->previous_crop
                        ]
                    ],
                    "model_path" => "output/best_hopular_model.pt",
                    "metadata_path" => "output/metadata.pkl"
                ]);

            if ($aiResponse->failed()) {
                throw new Exception("AI Service Error: " . $aiResponse->status());
            }

            $resultAI = $aiResponse->json();
            
            // Berdasarkan api.py Anda, response ada di dalam ['predictions'][0]
            $predictions = $resultAI['predictions'][0] ?? [];

            if (empty($predictions)) {
                throw new Exception("AI tidak memberikan hasil prediksi.");
            }

            // 5. Simpan hasil prediksi ke database
            $now = Carbon::now();
            $insertData = [];

            foreach ($predictions as $pred) {
                $insertData[] = [
                    "input_id" => $input_id,
                    "recommended_crop" => $pred['nama_tanaman'],
                    "care_instructions" => $pred['keterangan'] ?? null, 
                    "score" => $pred['kecocokan'] ?? null,
                    "created_at" => $now,
                    "updated_at" => $now
                ];
            }

            DB::table('crop_recommendation')->insert($insertData);

            // 6. Ambil ulang data yang baru disimpan untuk mapping ke FE
            $saved = DB::table('crop_recommendation')
                ->where('input_id', $input_id)
                ->get();

            $recommendations = $saved->map(function($item) use ($careManual) {
                return [
                    "nama_tanaman" => $item->recommended_crop,
                    "care_instructions" => $careManual[$item->recommended_crop] ?? ($item->care_instructions ?? "Instruksi tidak tersedia.")
                ];
            });

            return response()->json([
                "success" => true,
                "input_id" => $input_id,
                "recommendations" => $recommendations
            ]);

        } catch (Exception $e) {
            return response()->json([
                "success" => false,
                "message" => "Gagal memproses rekomendasi: " . $e->getMessage()
            ], 500);
        }
    }
}