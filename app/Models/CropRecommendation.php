<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class CropRecommendation extends Model
{
    protected $table = "crop_recommendation";
    protected $primaryKey = "recommendation_id";
    public $timestamps = false;

    protected $fillable = [
        "input_id",
        "recommended_crop",
        "care_instructions",
        "recommended_at",
    ];

    public function input()
    {
        return $this->belongsTo(Input::class, "input_id");
    }
}
