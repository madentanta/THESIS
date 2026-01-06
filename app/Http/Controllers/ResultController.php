<?php

namespace App\Http\Controllers;

use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Http;
use Carbon\Carbon;

class ResultController extends Controller
{
    /**
     * Ambil semua rekomendasi tanaman berdasarkan input_id
     * FE hanya menerima nama_tanaman & care_instructions hardcode
     * DB tetap menyimpan semua AI response (score & keterangan)
     */
    public function getRecommendation($input_id)
    {
 // Hardcode care instructions untuk FE (lebih lengkap)
$careManual = [
    "Tebu" => "1. Siram tiap 3 hari.\n2. Pupuk NPK tiap 2 minggu.\n3. Pastikan tanah gembur dan subur.\n4. Jaga jarak tanam 1.2 m antar tanaman.\n5. Panen setelah 10-12 bulan.",
    "Jagung" => "1. Siram tiap 2 hari, terutama saat musim kemarau.\n2. Pupuk urea atau NPK tiap 1 minggu.\n3. Tanam di tanah yang kaya organik.\n4. Pastikan mendapat sinar matahari penuh.\n5. Jaga jarak tanam 25-30 cm per tanaman.\n6. Panen setelah 3-4 bulan.",
    "Padi" => "1. Sirami rutin, jaga air tetap tergenang 5-10 cm.\n2. Pupuk NPK dan pupuk organik sesuai dosis.\n3. Kontrol hama seperti wereng dan penggerek batang.\n4. Pastikan padi mendapat sinar matahari cukup.\n5. Panen setelah 4-5 bulan."
];

        // 1. Cek apakah rekomendasi sudah ada di DB
        $existing = DB::table('crop_recommendation')
            ->where('input_id', $input_id)
            ->get();

        if ($existing->isNotEmpty()) {
            // Hanya ambil nama tanaman & care_instructions hardcode untuk response
            $recommendations = $existing->map(function($item) use ($careManual) {
                return [
                    "nama_tanaman" => $item->recommended_crop,
                    "care_instructions" => $careManual[$item->recommended_crop] ?? null
                ];
            });

            return response()->json([
                "success" => true,
                "input_id" => $input_id,
                "recommendations" => $recommendations
            ]);
        }

        // 2. Ambil input data
        $inputData = DB::table('input')->where('input_id', $input_id)->first();

        if (!$inputData) {
            return response()->json([
                "success" => false,
                "message" => "Input data tidak ditemukan."
            ], 404);
        }

        try {
            // 3. Call Hopular API
            $aiResponse = Http::timeout(15)
                ->retry(3, 1000)
                ->post("https://plantadvisor.cloud/inference", [
                    "input_data" => [
                        [
                            "soil_ph" => floatval($inputData->soil_ph),
                            "temperature" => floatval($inputData->temperature),
                            "humidity" => floatval($inputData->humidity),
                            "location" => $inputData->location,
                            "previous_crop" => $inputData->previous_crop
                        ]
                    ],
                    "model_path" => "best_hopular_model.pt",
                    "metadata_path" => "metadata.pkl"
                ]);

            if ($aiResponse->failed()) {
                throw new \Exception("Hopular API error: " . $aiResponse->body());
            }

            $resultAI = $aiResponse->json();

            // 4. Ambil semua prediksi
            $predictions = $resultAI['predictions'][0] ?? [];

            if (empty($predictions)) {
                throw new \Exception("Tidak ada rekomendasi dari AI");
            }

            // 5. Simpan semua ke DB (termasuk score & keterangan)
            $now = Carbon::now();
            $insertData = [];

            foreach ($predictions as $pred) {
                $insertData[] = [
                    "input_id" => $input_id,
                    "recommended_crop" => $pred['nama_tanaman'],
                    "care_instructions" => $pred['keterangan'] ?? null, // simpan AI keterangan
                    "score" => $pred['kecocokan'] ?? null,
                    "created_at" => $now,
                    "updated_at" => $now
                ];
            }

            DB::table('crop_recommendation')->insert($insertData);

            $saved = DB::table('crop_recommendation')
                ->where('input_id', $input_id)
                ->get();

            // Response FE hanya nama_tanaman & hardcode care_instructions
            $recommendations = $saved->map(function($item) use ($careManual) {
                return [
                    "nama_tanaman" => $item->recommended_crop,
                    "care_instructions" => $careManual[$item->recommended_crop] ?? null
                ];
            });

        } catch (\Exception $e) {
            return response()->json([
                "success" => false,
                "message" => "Gagal memproses rekomendasi: " . $e->getMessage()
            ], 500);
        }

        // 6. Return semua rekomendasi ke FE tanpa score
        return response()->json([
            "success" => true,
            "input_id" => $input_id,
            "recommendations" => $recommendations
        ]);
    }
}
