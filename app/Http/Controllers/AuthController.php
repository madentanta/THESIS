<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Hash;
use Illuminate\Support\Facades\Mail;
use Illuminate\Validation\ValidationException;
use Carbon\Carbon;

class AuthController extends Controller
{
    // REGISTER
    public function register(Request $req)
    {
        // ATURAN UNIQUE: 'unique:user,username' dan 'unique:user,email'
        $req->validate([
            "fullname" => "required",
            "username" => "required|unique:user,username",
            "email"    => "required|email|unique:user,email",
            "password" => [
                "required",
                "min:5",
                "regex:/[a-zA-Z]/",
                "regex:/[0-9]/",
                "regex:/[^a-zA-Z0-9]/"
            ]
        ]);

        // 1. BUAT TOKEN VERIFIKASI BARU
        $verificationToken = base64_encode(random_bytes(60)); 
        
        $id = DB::table("user")->insertGetId([
            "fullname"              => $req->fullname,
            "username"              => $req->username,
            "email"                 => $req->email,
            "password_hash"         => Hash::make($req->password),
            "verification_token"    => $verificationToken,    // SIMPAN TOKEN
            "email_verified_at"     => null,                 // BELUM TERVERIFIKASI
            "created_at"            => now(),
        ], 'user_id'); 

        //  KIRIM EMAIL VERIFIKASI
        // 
        $verificationLink = "https://plantadvisor.cloud/api/verify-email?token=" . urlencode($verificationToken);
        
        $emailBody = 
            "Halo,\n\n" .
            "Kami mencatat pendaftaran akun baru menggunakan alamat email ini. Jika Anda yang mendaftar, silakan klik tautan di bawah ini untuk mengaktifkan akun Anda:\n\n" .
            "------------------------------\n" .
            "[ Ya, Itu Saya - Aktifkan Akun ]\n" . 
            $verificationLink . "\n" .
            "------------------------------\n\n" .
            "Jika Anda tidak mendaftar, Anda dapat mengabaikan email ini.\n\n" .
            "Terima kasih,\n" .
            "Tim Plant Advisor";

        Mail::raw(
            $emailBody, 
            function ($m) use ($req) {
                $m->to($req->email)->subject("Verifikasi Email | Plant Advisor");
            }
        );

        return response()->json([
            "message" => "Registration successful. Please check your email to verify your account.",
            "user_id" => $id
        ], 201);
    }

    // LOGIN (Wajib Verifikasi Email)
   public function login(Request $req)
{
    $req->validate([
        "username" => "required",
        "password" => "required"
    ]);

    $user = DB::table("user")->where("username", $req->username)->first();

    // Jika user tidak ada → langsung 401 (tidak perlu kena ban)
    if (!$user) {
        return response()->json([
            "error" => "Unauthorized",
            "message" => "Invalid username or password"
        ], 401);
    }



if ($user->locked_until) {

    // --- Parse timestamp dari DB sebagai UTC lalu convert ke WIB ---
$lockedUntil = Carbon::parse($user->locked_until, 'UTC')->setTimezone('Asia/Jakarta');
$nowLocal    = Carbon::now('Asia/Jakarta');


    // Jika waktu lockout masih di masa depan
    if ($lockedUntil->isFuture()) {

        // Hitung sisa detik (selalu angka positif)
        $remainingSeconds = $nowLocal->diffInSeconds($lockedUntil);

        // Jika waktu habis → reset lockout
        if ($remainingSeconds <= 0) {

            DB::table("user")->where("user_id", $user->user_id)->update([
                "login_attempts" => 0,
                "locked_until" => null,
            ]);

            // lanjut proses password check

        } else {

            // Format untuk menit & detik
            $minutes = floor($remainingSeconds / 60);
            $seconds = $remainingSeconds % 60;

            $timeString = "";

            if ($minutes > 0) {
                $timeString .= "$minutes menit";
                if ($seconds > 0) $timeString .= " ";
            }

            if ($seconds > 0 || $minutes === 0) {
                $timeString .= "$seconds detik";
            }

            if ($timeString === "") {
                $timeString = "beberapa saat";
            }

            // Return response 423 (Locked)
            return response()->json([
                "error" => "Locked",
                "message" => "Akun diblokir sementara. Coba lagi dalam $timeString."
            ], 423);
        }

    } else {

        // Jika waktu lockout sudah lewat → reset
        DB::table("user")->where("user_id", $user->user_id)->update([
            "login_attempts" => 0,
            "locked_until" => null,
        ]);

        // lanjut proses password check
    }
}

// Jika locked_until NULL → lanjut proses password check


    // 🛑 VALIDASI PASSWORD
    if (!Hash::check($req->password, $user->password_hash)) {

        $attempts = $user->login_attempts + 1;

        // Jika sudah 3x → ban 5 menit
        if ($attempts >= 3) {
            DB::table("user")->where("user_id", $user->user_id)->update([
                "login_attempts" => 0,
                "locked_until"   => now()->addMinutes(5)
            ]);

            return response()->json([
                "error" => "Locked",
                "message" => "Akun diblokir selama 5 menit karena 3 kali salah login."
            ], 423);
        }

        // Jika masih < 3 → tambah attempt
        DB::table("user")->where("user_id", $user->user_id)->update([
            "login_attempts" => $attempts
        ]);

        return response()->json([
            "error" => "Unauthorized",
            "message" => "Invalid username or password (" . (3 - $attempts) . " kesempatan tersisa)"
        ], 401);
    }

    // reset attempts
    DB::table("user")->where("user_id", $user->user_id)->update([
        "login_attempts" => 0,
        "locked_until" => null,
    ]);

    // Cek verifikasi email
    if ($user->email_verified_at === null) {
        return response()->json([
            "error" => "Email not verified",
            "message" => "Akun Anda belum aktif. Mohon verifikasi email Anda."
        ], 403);
    }

    // Generate Auth Token
    $token = base64_encode(random_bytes(40));
    DB::table("user")->where("user_id", $user->user_id)->update([
        "auth_token" => $token
    ]);

    return response()->json([
        "message" => "Login success",
        "token"   => $token,
        "user_id" => $user->user_id
    ]);
}


    // LOGOUT
    public function logout(Request $req)
    {
        $token = $req->header("Authorization");

        DB::table("user")
            ->where("auth_token", $token)
            ->update(["auth_token" => null]);

        return response()->json(["message" => "Logout success"]);
    }


    // VERIFY EMAIL
    public function verifyEmail(Request $req)
    {
        $req->validate(["token" => "required"]);
        
        $loginUrl = 'https://plantadvisor.cloud/login.html'; // URL tujuan redirect (Frontend)

        $user = DB::table("user")
                  ->where("verification_token", $req->token)
                  ->whereNull('email_verified_at')
                  ->first();

        // 1. Penanganan Error / Sudah Terverifikasi
        if (!$user) {
            $verifiedUser = DB::table("user")->where("verification_token", $req->token)->whereNotNull('email_verified_at')->first();
            
            $redirectStatus = $verifiedUser ? 'already' : 'invalid_token';
            $message = $verifiedUser ? 'Akun Anda sudah diverifikasi. Mengalihkan ke halaman login...' : 'Verifikasi gagal. Tautan tidak valid. Mengalihkan...';
            $color = $verifiedUser ? '#4CAF50' : '#FF4D4D';
            
            // Redirect setelah 3 detik
            return $this->generateLoadingPage($message, $color, $loginUrl . '?verified=' . $redirectStatus);
        }

        // 2. Verifikasi Sukses
        DB::table("user")->where("user_id", $user->user_id)->update([
            "email_verified_at"  => now(),
            "verification_token" => null 
        ]);

        $message = 'Verifikasi email berhasil! Akun Anda telah diaktifkan. Mengalihkan ke halaman login...';
        
        // Redirect setelah 3 detik
        return $this->generateLoadingPage($message, '#4CAF50', $loginUrl . '?verified=true');
    }

    // ⭐ FUNGSI HELPER BARU UNTUK MEMBUAT RESPONS LOADING HTML ⭐
    private function generateLoadingPage(string $message, string $color, string $redirectUrl)
    {
        // Meta refresh akan mengalihkan browser ke $redirectUrl setelah 3 detik
        $htmlResponse = '
            <!DOCTYPE html>
            <html lang="id">
            <head>
                <title>Memproses Verifikasi</title>
                <meta http-equiv="refresh" content="3;url='. $redirectUrl .'"> 
                <style>
                    body { margin: 0; display: flex; justify-content: center; align-items: center; min-height: 100vh; background-color: #f0f0f0; font-family: sans-serif; }
                    .container { text-align: center; padding: 40px; background: white; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
                    .spinner { border: 8px solid #f3f3f3; border-top: 8px solid '. $color .'; border-radius: 50%; width: 60px; height: 60px; animation: spin 2s linear infinite; margin: 0 auto 20px; }
                    .message { color: #333; font-size: 1.1em; }
                    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="spinner"></div>
                    <h1>Memproses Verifikasi Akun</h1>
                    <p class="message">'. $message .'</p>
                </div>
            </body>
            </html>
        ';
        
        return response($htmlResponse, 200)->header('Content-Type', 'text/html');
    }


    // FORGOT PASSWORD (Aman: Wajib Email Terverifikasi)
public function forgotPassword(Request $req)
{
    $req->validate(["email" => "required|email"]);

    $user = DB::table("user")
                ->where("email", $req->email)
                ->whereNotNull('email_verified_at')
                ->first();

    if (!$user) {
        throw ValidationException::withMessages([
            "email" => ["Email not found or not yet verified."]
        ]);
    }

    // 1. Generate raw token (dipakai user di email)
    $rawToken = base64_encode(random_bytes(50));

    // 2. Hash token dengan SHA256 untuk disimpan ke DB
    $hashedToken = hash("sha256", $rawToken);

    // 3. Simpan hashed token ke database
    DB::table("user")->where("user_id", $user->user_id)->update([
        "reset_token"         => $hashedToken,
        "reset_token_created" => now()
    ]);

    // 4. Link reset (raw token, email tetap sama)
    $resetLink = "https://plantadvisor.cloud/reset-password.html?token=" . urlencode($rawToken);

    // 5. Email tetap sama persis
    $emailBody = 
        "Halo,\n\n" .
        "Kami menerima permintaan untuk mereset password akun Plant Advisor Anda.\n\n" .
        "Silakan klik tautan di bawah ini untuk mengatur password baru Anda:\n\n" .
        $resetLink . "\n\n" .
        "Tautan ini hanya berlaku untuk waktu terbatas (3 menit).\n\n" . 
        "Jika Anda tidak mengajukan permintaan ini, Anda dapat mengabaikan email ini dan password Anda akan tetap aman.\n\n" .
        "Terima kasih,\n\n" .
        "Tim Plant Advisor";

    Mail::raw(
        $emailBody, 
        function ($m) use ($req) {
            $m->to($req->email)->subject("Reset Password | Plant Advisor");
        }
    );

    return ["message" => "Reset password email sent"];
}



    // RESET PASSWORD
public function resetPassword(Request $req)
{
    $req->validate([
        "token" => "required",
        "password" => [
            "required",
            "min:5",
            "regex:/[a-zA-Z]/",
            "regex:/[0-9]/",
            "regex:/[^a-zA-Z0-9]/"
        ]
    ]);

    // 1. Hash token yang diterima dari user
    $hashedToken = hash("sha256", $req->token);

    // 2. Cocokkan dengan DB
    $user = DB::table("user")->where("reset_token", $hashedToken)->first();

    if (!$user) {
        return response()->json(["error" => "Invalid token"], 400);
    }

    // 3. Cek kedaluwarsa (TTL = 3 menit)
    if (Carbon::parse($user->reset_token_created)->addMinutes(3)->lt(now())) {

        // Hapus token expired
        DB::table("user")->where("user_id", $user->user_id)->update([
            "reset_token" => null,
            "reset_token_created" => null
        ]);

        return response()->json(["error" => "Token expired"], 400);
    }

    // 4. Password baru tidak boleh sama dengan lama
    if (Hash::check($req->password, $user->password_hash)) {
        return response()->json([
            "error" => "Security Policy Violation",
            "message" => "Password baru tidak boleh sama dengan password lama."
        ], 400);
    }

    // 5. Update password & hapus token (single use)
    DB::table("user")->where("user_id", $user->user_id)->update([
        "password_hash"         => Hash::make($req->password),
        "reset_token"           => null,
        "reset_token_created"   => null
    ]);

    // 6. Notifikasi email (tetap sama)
    Mail::raw(
        "Halo,\n\n" .
        "Password akun Plant Advisor Anda telah berhasil diubah.\n\n" .
        "Jika Anda melakukan perubahan ini, Anda dapat mengabaikan email ini.\n\n" .
        "Jika Anda TIDAK melakukan perubahan ini, segera hubungi layanan pelanggan kami atau coba reset password Anda lagi.\n\n" .
        "Terima kasih,\n\n" .
        "Tim Plant Advisor",
        function ($m) use ($user) {
            $m->to($user->email)->subject("Notifikasi: Password Berhasil Diubah");
        }
    );

    return ["message" => "Password reset successfully"];
}

    
    // GET USER STATUS (Optional, untuk mengecek status login/verifikasi)
    public function getUserStatus(Request $req)
    {
        $user = $req->user(); 
        
        if (!$user) {
             return response()->json(["error" => "User not authenticated"], 401);
        }

        return response()->json([
            "user_id" => $user->user_id,
            "username" => $user->username,
            "email_verified" => $user->email_verified_at !== null,
        ]);
    }
}