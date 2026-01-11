<?php

namespace App\Http\Controllers;

use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Http;
use Carbon\Carbon;
use Exception;

class ResultController extends Controller
{
    /**
     * Ambil semua rekomendasi tanaman berdasarkan input_id
     * Terintegrasi dengan FastAPI (Hopular)
     */
    public function getRecommendation($input_id)
    {
        // 1. Manual perawatan (stabil untuk FE)
        $careManual = [
            "Tebu" => "1. Siram tiap 3 hari.\n2. Pupuk NPK tiap 2 minggu.\n3. Pastikan tanah gembur dan subur.\n4. Jaga jarak tanam 1.2 m antar tanaman.\n5. Panen setelah 10-12 bulan.",
            "Jagung" => "1. Siram tiap 2 hari.\n2. Pupuk urea/NPK tiap 1 minggu.\n3. Sinar matahari penuh.\n4. Jarak tanam 25-30 cm.\n5. Panen 3-4 bulan.",
            "Padi" => "1. Air tergenang 5–10 cm.\n2. Pupuk sesuai dosis.\n3. Kendalikan hama.\n4. Panen 4–5 bulan."
        ];

        // 2. Ambil cache hasil rekomendasi (ANTI HIT AI ULANG)
        $existing = DB::table('crop_recommendation')
            ->where('input_id', $input_id)
            ->get();

        if ($existing->isNotEmpty()) {
            return response()->json([
                "success" => true,
                "input_id" => $input_id,
                "recommendations" => $existing->map(function ($item) use ($careManual) {
                    return [
                        "nama_tanaman" => $item->recommended_crop,
                        "care_instructions" =>
                            $careManual[$item->recommended_crop]
                            ?? $item->care_instructions
                            ?? "Instruksi tidak tersedia."
                    ];
                })
            ]);
        }

        // 3. Ambil data input
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
            // 4. Call AI Hopular
            $aiResponse = Http::timeout(60)
                ->retry(2, 1000)
                ->post("http://ai:8001/inference", [
                    "input_data" => [[
                        "soil_ph"        => (float) $inputData->soil_ph,
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

            $resultAI = $aiResponse->json();
            $rawPredictions = $resultAI['predictions'][0] ?? [];

            // Handle string JSON / array
            $predictions = is_string($rawPredictions)
                ? json_decode($rawPredictions, true)
                : $rawPredictions;

            if (empty($predictions)) {
                throw new Exception("AI tidak memberikan hasil prediksi.");
            }

            // 5. Simpan hasil AI (TERMASUK score, TAPI TIDAK KE FE)
            $now = Carbon::now();
            $insertData = [];

            foreach ($predictions as $pred) {
                $insertData[] = [
                    "input_id"          => $input_id,
                    "recommended_crop"  => $pred['nama_tanaman'],
                    "care_instructions" => $pred['keterangan'] ?? null,
                    "score"             => $pred['kecocokan'] ?? null,
                    "recommended_at"    => $now
                ];
            }

            DB::table('crop_recommendation')->insert($insertData);

            // 6. Return response ke FE (TANPA SCORE)
            return response()->json([
                "success" => true,
                "input_id" => $input_id,
                "recommendations" => collect($insertData)->map(function ($item) use ($careManual) {
                    return [
                        "nama_tanaman" => $item['recommended_crop'],
                        "care_instructions" =>
                            $careManual[$item['recommended_crop']]
                            ?? $item['care_instructions']
                            ?? "Instruksi tidak tersedia."
                    ];
                })
            ]);

        } catch (Exception $e) {
            return response()->json([
                "success" => false,
                "message" => "Gagal memproses rekomendasi: " . $e->getMessage()
            ], 500);
        }
    }
}
