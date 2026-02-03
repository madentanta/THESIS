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
    "Padi" =>
        "Padi adalah tanaman pangan penghasil beras yang tumbuh optimal di lahan sawah " .
        "dengan kondisi air tergenang. Tanaman ini membutuhkan pengelolaan air yang stabil, " .
        "pemupukan teratur, serta pengendalian hama agar menghasilkan gabah berkualitas " .
        "dalam waktu panen sekitar 4–5 bulan.",

    "Tebu" =>
        "Tebu merupakan tanaman perkebunan yang menjadi bahan baku utama pembuatan gula. " .
        "Tanaman ini membutuhkan tanah yang gembur dan subur, jarak tanam yang cukup lebar, " .
        "serta pemupukan rutin. Dengan perawatan yang baik, tebu dapat dipanen setelah " .
        "10–12 bulan masa tanam.",

    "Jagung" =>
        "Jagung adalah tanaman palawija yang mudah dibudidayakan dan membutuhkan sinar " .
        "matahari penuh. Tanaman ini memerlukan penyiraman rutin, pemupukan berkala, " .
        "serta pengaturan jarak tanam agar pertumbuhan optimal dan dapat dipanen dalam " .
        "waktu 3–4 bulan."
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
            $aiResponse = Http::timeout(60)->retry(2, 1000)->post("https://ai:8001/inference", [
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