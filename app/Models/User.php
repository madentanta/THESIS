<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class User extends Model
{
    protected $table = "user";
    protected $primaryKey = "user_id";
    public $timestamps = false;

    protected $fillable = [
        "fullname",
        "username",
        "email",
        "password_hash",
        "created_at",
        "auth_token",
        "reset_token",
        "reset_token_created",
    ];

    public function inputs()
    {
        return $this->hasMany(Input::class, "user_id");
    }
}
