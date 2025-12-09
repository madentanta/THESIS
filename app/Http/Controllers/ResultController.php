<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;
use Carbon\Carbon;

class ResultController extends Controller
{
    public function getRecommendation($input_id)
    {
        $statusCode = 200;

        // 1. Cek apakah rekomendasi sudah ada
        $recommendation = DB::table('crop_recommendation')
                            ->where('input_id', $input_id)
                            ->first();

        // 2. Jika belum ada → generate baru
        if (!$recommendation) {

            // Ambil input berdasarkan input_id
            $inputData = DB::table('input')->where('input_id', $input_id)->first();

            if (!$inputData) {
                return response()->json([
                    "message" => "Input data tidak ditemukan untuk ID: " . $input_id
                ], 404);
            }

            try {
                // Simulasi AI
                $resultAI = $this->simulateAI($inputData);

                // Encode JSON ke text
                $jsonResult = json_encode($resultAI);

                // FIX: PostgreSQL tidak punya kolom 'id', pakai returning()
                $inserted = DB::table("crop_recommendation")
                    ->insertGetId([
                        "input_id"          => $input_id,
                        "recommended_crop"  => $jsonResult,
                        "care_instructions" => null,
                        "recommended_at"    => Carbon::now(),
                    ], "recommendation_id"); // <-- FIX INI

                // Ambil ulang data
                $recommendation = DB::table('crop_recommendation')
                                    ->where('recommendation_id', $inserted)
                                    ->first();

                $statusCode = 201;

            } catch (\Exception $e) {
                return response()->json([
                    "message" => "Gagal memproses rekomendasi AI: " . $e->getMessage()
                ], 500);
            }
        }

        // 3. Return dengan nama kolom yang benar
        return response()->json([
            "message" => $statusCode == 201
                ? "Recommendation generated and retrieved successfully"
                : "Recommendation data retrieved from cache successfully",
            "recommendation_id"   => $recommendation->recommendation_id,
            "recommendation_data" => json_decode($recommendation->recommended_crop, true),
        ], $statusCode);
    }

    /**
     * Simulasi AI
     */
    private function simulateAI($inputData)
    {
        $ph = $inputData->soil_ph;
        $temp = $inputData->temperature;

        $recs = [];

        if ($ph >= 6.0 && $ph <= 7.5 && $temp >= 20 && $temp <= 30) {
            $recs[] = [
                'nama_tanaman' => 'Padi Sawah',
                'kecocokan'    => 0.95,
                'keterangan'   => 'PH optimal dan suhu ideal untuk padi. Pastikan pengairan stabil.'
            ];
            $recs[] = [
                'nama_tanaman' => 'Cabai Merah',
                'kecocokan'    => 0.88,
                'keterangan'   => 'Cocok, tanah harus memiliki drainase yang baik.'
            ];
        } else if ($ph < 6.0) {
            $recs[] = [
                'nama_tanaman' => 'Ubi Jalar',
                'kecocokan'    => 0.75,
                'keterangan'   => 'Toleransi terhadap pH asam. Pastikan kelembaban cukup.'
            ];
        } else {
            $recs[] = [
                'nama_tanaman' => 'Jagung Hibrida',
                'kecocokan'    => 0.65,
                'keterangan'   => 'Dapat beradaptasi pada suhu lebih tinggi, perlu cek kesuburan NPK.'
            ];
        }

        return $recs;
    }
}
