<?php

namespace App\Http\Controllers;

use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Http;
use Carbon\Carbon;

class ResultController extends Controller
{
    /**
     * Ambil semua rekomendasi tanaman berdasarkan input_id
     */
    public function getRecommendation($input_id)
    {
        // 1. Cek apakah rekomendasi sudah ada
        $existing = DB::table('crop_recommendation')
            ->where('input_id', $input_id)
            ->get();

        if ($existing->isNotEmpty()) {
            // Hanya ambil nama tanaman dan care_instructions untuk response
            $recommendations = $existing->map(function($item) {
                return [
                    "nama_tanaman" => $item->recommended_crop,
                    "care_instructions" => $item->care_instructions
                ];
            });

            return response()->json([
                "success" => true,
                "input_id" => $input_id,
                "recommendations" => $recommendations
            ]);
        }

        // 2. Ambil input data
        $inputData = DB::table('input')
            ->where('input_id', $input_id)
            ->first();

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
                            "soil_ph"       => floatval($inputData->soil_ph),
                            "temperature"   => floatval($inputData->temperature),
                            "humidity"      => floatval($inputData->humidity),
                            "location"      => $inputData->location,
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

            // 4. Ambil SEMUA prediksi
            $predictions = $resultAI['predictions'][0] ?? [];

            if (empty($predictions)) {
                throw new \Exception("Tidak ada rekomendasi dari AI");
            }

            // 5. Simpan SEMUA ke DB (termasuk score/kecocokan)
            $now = Carbon::now();
            $insertData = [];

            foreach ($predictions as $pred) {
                $insertData[] = [
                    "input_id" => $input_id,
                    "recommended_crop" => $pred['nama_tanaman'],
                    "care_instructions" => $pred['keterangan'] ?? null,
                    "score" => $pred['kecocokan'] ?? null, // tetap disimpan di DB
                    "created_at" => $now,
                    "updated_at" => $now
                ];
            }

            DB::table('crop_recommendation')->insert($insertData);

            $saved = DB::table('crop_recommendation')
                ->where('input_id', $input_id)
                ->get();

            // Hanya ambil nama tanaman & care_instructions untuk response
            $recommendations = $saved->map(function($item) {
                return [
                    "nama_tanaman" => $item->recommended_crop,
                    "care_instructions" => $item->care_instructions
                ];
            });

        } catch (\Exception $e) {
            return response()->json([
                "success" => false,
                "message" => "Gagal memproses rekomendasi: " . $e->getMessage()
            ], 500);
        }

        // 6. Return semua rekomendasi tanpa score/kecocokan
        return response()->json([
            "success" => true,
            "input_id" => $input_id,
            "recommendations" => $recommendations
        ]);
    }
}
