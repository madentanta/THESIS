<?php

namespace App\Http\Controllers;

use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Http;
use Carbon\Carbon;
use Exception;

class ResultController extends Controller
{
    public function getRecommendation($input_id)
    {
        $careManual = [
            "Tebu" => "1. Siram tiap 3 hari.\n2. Pupuk NPK tiap 2 minggu.\n3. Pastikan tanah gembur dan subur.\n4. Jaga jarak tanam 1.2 m antar tanaman.\n5. Panen setelah 10-12 bulan.",
            "Jagung" => "1. Siram tiap 2 hari.\n2. Pupuk urea/NPK tiap 1 minggu.\n3. Sinar matahari penuh.\n4. Jarak tanam 25-30 cm.\n5. Panen 3-4 bulan.",
            "Padi" => "1. Air tergenang 5–10 cm.\n2. Pupuk sesuai dosis.\n3. Kendalikan hama.\n4. Panen 4–5 bulan."
        ];

        // 1. Cek Cache Database
        $existing = DB::table('crop_recommendation')->where('input_id', $input_id)->get();

        if ($existing->isNotEmpty()) {
            return response()->json([
                "success" => true,
                "input_id" => (int) $input_id,
                "recommendations" => $existing->map(function ($item) use ($careManual) {
                    return [
                        "nama_tanaman" => $item->recommended_crop,
                        "care_instructions" => $careManual[$item->recommended_crop] ?? $item->care_instructions ?? "Instruksi tidak tersedia."
                    ];
                })
            ]);
        }

        // 2. Ambil Input Data
        $inputData = DB::table('input')->where('input_id', $input_id)->first();
        if (!$inputData) {
            return response()->json(["success" => false, "message" => "Input data tidak ditemukan."], 404);
        }

        try {
            // 3. Panggil AI
            $aiResponse = Http::timeout(60)->retry(2, 1000)->post("http://ai:8001/inference", [
                "input_data" => [[
                    "soil_ph"       => (float) $inputData->soil_ph,
                    "temperature"   => (float) $inputData->temperature,
                    "humidity"      => (float) $inputData->humidity,
                    "location"      => $inputData->location,
                    "previous_crop" => $inputData->previous_crop
                ]],
                "model_path"    => "output/best_hopular_model.pt",
                "metadata_path" => "output/metadata.pkl"
            ]);

            if ($aiResponse->failed()) {
                throw new Exception("AI Service Error: " . $aiResponse->status());
            }

            $responseBody = $aiResponse->json();
            $prediction = null;

            // --- PERBAIKAN BERDASARKAN LOG KAMU ---
            // Data kamu ada di: $responseBody['predictions'][0]
            if (isset($responseBody['predictions'][0]['nama_tanaman'])) {
                $prediction = $responseBody['predictions'][0];
            } elseif (isset($responseBody[0]['nama_tanaman'])) {
                $prediction = $responseBody[0];
            }

            if (!$prediction) {
                throw new Exception("Format AI tidak valid.");
            }
            // --- END PERBAIKAN ---

            $cropName = $prediction['nama_tanaman'];
            $score = $prediction['kecocokan'] ?? 0;
            $now = Carbon::now();

            // 4. Simpan ke Database
            DB::table('crop_recommendation')->insert([
                "input_id"          => $input_id,
                "recommended_crop"  => $cropName,
                "care_instructions" => $careManual[$cropName] ?? "Instruksi tidak tersedia.",
                "score"             => $score,
                "recommended_at"    => $now
            ]);

            // 5. Response ke Front-End
            return response()->json([
                "success" => true,
                "input_id" => (int) $input_id,
                "recommendations" => [
                    [
                        "nama_tanaman" => $cropName,
                        "care_instructions" => $careManual[$cropName] ?? "Instruksi tidak tersedia."
                    ]
                ]
            ]);

        } catch (Exception $e) {
            return response()->json([
                "success" => false,
                "message" => "Gagal memproses rekomendasi: " . $e->getMessage()
            ], 500);
        }
    }
}