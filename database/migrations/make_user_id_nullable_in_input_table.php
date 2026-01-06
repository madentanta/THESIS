<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::table('input', function (Blueprint $table) {
            // Mengubah kolom user_id agar dapat menerima nilai NULL
            // Pastikan Anda sudah menginstall composer require doctrine/dbal jika belum
            $table->integer('user_id')->nullable()->change();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('input', function (Blueprint $table) {
            // Mengembalikan kolom user_id menjadi NOT NULL (Wajib diisi)
            $table->integer('user_id')->nullable(false)->change();
        });
    }
};