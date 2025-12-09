<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Jalankan migrasi (Menambahkan kolom).
     */
    public function up(): void
    {
        // Pastikan nama tabel adalah 'user' (tunggal)
        Schema::table('user', function (Blueprint $table) {
            // Menambahkan kolom auth_token dengan tipe string yang cukup panjang (70 karakter),
            // boleh null (karena token belum ada saat register) dan harus unik.
            $table->string('auth_token', 70)->nullable()->unique()->after('email');
        });
    }

    /**
     * Mengembalikan migrasi (Menghapus kolom).
     */
    public function down(): void
    {
        Schema::table('user', function (Blueprint $table) {
            $table->dropColumn('auth_token');
        });
    }
};