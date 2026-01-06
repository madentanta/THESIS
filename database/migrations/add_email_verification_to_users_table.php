<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::table('user', function (Blueprint $table) {
            // Kolom untuk menyimpan token verifikasi (digunakan untuk link verifikasi)
            // String dengan panjang 80 sudah cukup. Diletakkan setelah password_hash
            $table->string('verification_token', 80)->nullable()->after('password_hash'); 
            
            // Kolom untuk menyimpan timestamp kapan email diverifikasi
            // Jika NULL, berarti belum diverifikasi.
            $table->timestamp('email_verified_at')->nullable()->after('verification_token');
        });
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::table('user', function (Blueprint $table) {
            // Menghapus kolom jika rollback/down
            $table->dropColumn('verification_token');
            $table->dropColumn('email_verified_at');
        });
    }
};