<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Input extends Model
{
    protected $table = "input";
    protected $primaryKey = "input_id";
    public $timestamps = false;

    protected $fillable = [
        "user_id",
        "soil_ph",
        "location",
        "temperature",
        "humidity",
        "previous_crop",
        "submitted_at",
    ];

    public function user()
    {
        return $this->belongsTo(User::class, "user_id");
    }

    public function recommendation()
    {
        return $this->hasOne(CropRecommendation::class, "input_id");
    }
}
