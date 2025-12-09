<?php

namespace App\Http\Middleware;

use Closure;
use Illuminate\Http\Request;
use Symfony\Component\HttpFoundation\Response;

class BasicAuthAdmin
{
    /**
     * Handle an incoming request.
     *
     * @param  \Closure(\Illuminate\Http\Request): (\Symfony\Component\HttpFoundation\Response)  $next
     */
    public function handle(Request $request, Closure $next): Response
    {
        // 1. Ambil kredensial yang valid dari .env
        $validUsername = env('API_ADMIN_USERNAME');
        $validPassword = env('API_ADMIN_PASSWORD');

        // 2. Ambil kredensial yang dikirim melalui Basic Auth header
        $username = $request->getUser();
        $password = $request->getPassword();

        // Cek jika kredensial dari request tidak ada
        if (empty($username) || empty($password)) {
            return $this->unauthorizedResponse();
        }

        // 3. Verifikasi kredensial
        // Menggunakan hash_equals() jika Anda membandingkan string yang sensitif terhadap waktu, 
        // tetapi perbandingan langsung juga umum untuk Basic Auth sederhana.
        if ($username === $validUsername && $password === $validPassword) {
            return $next($request); // Kredensial cocok, lanjutkan request
        }

        // 4. Kredensial tidak cocok
        return $this->unauthorizedResponse(); 
    }

    /**
     * Mengembalikan respons 401 Unauthorized.
     * Termasuk header WWW-Authenticate untuk meminta kredensial Basic Auth.
     */
    protected function unauthorizedResponse(): Response
    {
        return response()->json([
            'status' => 'error',
            'message' => 'Autentikasi gagal. Kredensial Basic Auth tidak valid.'
        ], 401, ['WWW-Authenticate' => 'Basic']); // 401 Unauthorized
    }
}