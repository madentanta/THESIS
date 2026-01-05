<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class DatasetTanaman extends Model
{
    protected $table = "dataset_tanaman";
    protected $primaryKey = "id";
    public $timestamps = false;

    protected $fillable = [
        "nama_daerah",
        "fertility",
        "moisture",
        "ph",
        "temp",
        "sunlight",
        "humidity",
        "kecamatan",
        "nama_tanaman",
        "created_at",
    ];
}
