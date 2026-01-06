<?php

namespace App\Http\Controllers;

use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Http;
use Carbon\Carbon;

class ResultController extends Controller
{
    /**
     * Ambil rekomendasi tanaman berdasarkan input_id
     * Jika belum ada, generate 1 rekomendasi dari AI service / logic
     */
    public function getRecommendation($input_id)
    {
        // 1. Cek apakah rekomendasi sudah ada
        $recommendation = DB::table('crop_recommendation')
                            ->where('input_id', $input_id)
                            ->first();

        if (!$recommendation) {
            // Ambil input
            $inputData = DB::table('input')->where('input_id', $input_id)->first();

            if (!$inputData) {
                return response()->json([
                    "success" => false,
                    "message" => "Input data tidak ditemukan."
                ], 404);
            }

            try {
                // send ke ai
                $aiResponse = Http::post("http://ai-service:5000/predict", [
                    "soil_ph"       => floatval($inputData->soil_ph),
                    "temperature"   => floatval($inputData->temperature),
                    "humidity"      => floatval($inputData->humidity),
                    "location"      => $inputData->location,
                    "previous_crop" => $inputData->previous_crop
                ]);

                if ($aiResponse->failed()) {
                    throw new \Exception("AI Service error: " . $aiResponse->body());
                }

                $resultAI = $aiResponse->json();

                $insertedId = DB::table('crop_recommendation')
                    ->insertGetId([
                        "input_id"          => $input_id,
                        "recommended_crop"  => $resultAI['nama_tanaman'],
                        "care_instructions" => $resultAI['care_instructions'] ?? null
                    ], "recommendation_id");

                $recommendation = DB::table('crop_recommendation')
                                    ->where('recommendation_id', $insertedId)
                                    ->first();

            } catch (\Exception $e) {
                return response()->json([
                    "success" => false,
                    "message" => "Gagal memproses rekomendasi: " . $e->getMessage()
                ], 500);
            }
        }

        return response()->json([
            "success" => true,
            "input_id" => $input_id,
            "recommended_crop" => $recommendation->recommended_crop,
            "care_instructions" => $recommendation->care_instructions
        ]);
    }
}
