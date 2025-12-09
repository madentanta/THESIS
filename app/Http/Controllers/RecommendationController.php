<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;

class RecommendationController extends Controller
{
    public function generate(Request $req)
    {
        $req->validate([
            "input_id" => "required|numeric"
        ]);

        $crop = "Cabai";
        $instructions = "Cabai cocok dengan pH 5.5 - 7 dan suhu 18-30°C.";

        $id = DB::table("crop_recommendation")->insertGetId([
            "input_id" => $req->input_id,
            "recommended_crop" => $crop,
            "care_instructions" => $instructions
        ]);

        return [
            "message" => "Recommendation generated",
            "recommendation_id" => $id,
            "crop" => $crop,
            "instructions" => $instructions
        ];
    }
}
