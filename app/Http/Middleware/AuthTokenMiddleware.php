<?php

namespace App\Http\Middleware;

use Closure;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;
use Symfony\Component\HttpFoundation\Response;

// Pastikan nama kelasnya adalah AuthTokenMiddleware
class AuthTokenMiddleware
{
    /**
     * Handle an incoming request.
     *
     * @param  \Closure(\Illuminate\Http\Request): (\Symfony\Component\HttpFoundation\Response)  $next
     */
    public function handle(Request $req, Closure $next): Response
    {
        $token = $req->header('Authorization');

        if (!$token) {
            return response()->json(["error" => "Unauthorized: Token missing"], 401);
        }

        // *** PENTING: Membersihkan string "Bearer " ***
        // Token yang dikirim klien biasanya dalam format "Bearer [token_sebenarnya]".
        // Kita harus menghapus "Bearer " agar yang dicari di DB hanya token aslinya.
        $token = str_replace('Bearer ', '', $token);

        // Cari user di database
        $user = DB::table("user")->where("auth_token", $token)->first();

        if (!$user) {
            return response()->json(["error" => "Invalid token"], 401);
        }

        // Inject objek user ke request.
        // Data ini bisa diakses di Controller menggunakan $request->attributes->get("auth_user")
        $req->attributes->set("auth_user", $user);

        return $next($req);
    }
}