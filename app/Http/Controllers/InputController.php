<?php

namespace App\Http\Controllers;

use App\Models\Input;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Validator;
use Illuminate\Support\Facades\Auth; // Digunakan untuk Auth::id()

class InputController extends Controller
{
    /**
     * STORE INPUT
     * --------------------
     * Endpoint: POST /api/input/store
     */
    public function store(Request $request)
    {
        // 1. VALIDASI DATA
        $validator = Validator::make($request->all(), [
            // Batasan PH (0-14) lebih akurat, meskipun rentang 3-14 mungkin cukup untuk tanah.
            "soil_ph"       => "required|numeric|min:0|max:14",
            "location"      => "required|string|max:255",
            // Batasan suhu/kelembapan ditambahkan untuk menghindari nilai ekstrem/sampah.
            "temperature"   => "required|numeric|min:-50|max:100", // Range realistis
            "humidity"      => "required|numeric|min:0|max:100",  // Harus antara 0% dan 100%
            "previous_crop" => "nullable|string|max:255",
        ]);

        if ($validator->fails()) {
            return response()->json([
                "success" => false,
                "message" => "Validasi gagal",
                "errors"  => $validator->errors()
            ], 422);
        }

        try {
            // 2. TENTUKAN USER ID
            // Menggunakan Auth::id() yang lebih disukai daripada auth()->id() untuk konsistensi.
            // Jika user tidak login/tidak terautentikasi, id akan bernilai null.
            $userId = Auth::id();

            // 3. INSERT DATA
            // Gunakan $request->validated() untuk memastikan hanya data yang tervalidasi yang di-insert.
            $validatedData = $validator->validated();
            
            // Tambahkan data non-validated ke array
            $validatedData['user_id'] = $userId;
            $validatedData['submitted_at'] = now();

            $input = Input::create($validatedData);

            return response()->json([
                "success" => true,
                "message" => "Input berhasil disimpan",
                // Mengembalikan hanya ID input yang baru
                "input_id" => $input->input_id, 
                "data"    => $input
            ], 201);

        } catch (\Throwable $e) {
            // Log error untuk debugging di sisi server
            \Log::error("Gagal menyimpan data input: " . $e->getMessage(), ['exception' => $e]);
            
            // Menghapus detail error sensitif (line, file) dari response 500
            return response()->json([
                "success" => false,
                "message" => "Terjadi kesalahan server saat menyimpan data."
            ], 500);
        }
    }

    /**
     * LIST ALL INPUT
     * --------------------
     * Endpoint: GET /api/input
     * CATATAN: Fungsi ini HANYA boleh diakses oleh Admin atau harus difilter berdasarkan user_id.
     */
    public function index()
    {
        try {
            // Perbaikan: Batasi data hanya untuk user yang login, jika bukan admin.
            // Contoh implementasi sederhana (Jika user login, tampilkan data miliknya):
            $userId = Auth::id();
            
            if ($userId) {
                $data = Input::where('user_id', $userId)
                             ->orderBy("submitted_at", "DESC") // Urutkan berdasarkan waktu submit
                             ->get();
            } else {
                // Jika tidak login, atau ini adalah endpoint admin (perlu otorisasi lebih lanjut)
                // Jika ini adalah endpoint publik, Anda perlu mempertimbangkan apakah semua data boleh dilihat.
                // Jika ini endpoint admin, Anda perlu policy/middleware.
                $data = Input::orderBy("submitted_id", "DESC")->get();
            }

            return response()->json([
                "success" => true,
                "data"    => $data
            ]);
        } catch (\Throwable $e) {
             \Log::error("Gagal mengambil data input: " . $e->getMessage(), ['exception' => $e]);
            return response()->json([
                "success" => false,
                "message" => "Terjadi kesalahan server saat mengambil data."
            ], 500);
        }
    }

    /**
     * DETAIL INPUT
     * --------------------
     * Endpoint: GET /api/input/{id}
     * CATATAN: Tambahkan otorisasi untuk memastikan user hanya bisa melihat data miliknya.
     */
    public function show($id)
    {
        // 1. Cari data
        $data = Input::find($id);

        if (!$data) {
            return response()->json([
                "success" => false,
                "message" => "Input tidak ditemukan"
            ], 404);
        }

        // 2. Otorisasi (Hanya yang memiliki data atau admin yang boleh melihat)
        if ($data->user_id && Auth::id() !== $data->user_id) {
            // Anda bisa menggunakan 403 Forbidden, atau 404 Not Found (untuk keamanan)
            return response()->json([
                "success" => false,
                "message" => "Akses ditolak atau Input tidak ditemukan"
            ], 403); 
        }

        return response()->json([
            "success" => true,
            "data"    => $data
        ]);
    }

    /**
     * DELETE INPUT
     * --------------------
     * Endpoint: DELETE /api/input/{id}
     * CATATAN: Tambahkan otorisasi untuk memastikan user hanya bisa menghapus data miliknya.
     */
    public function delete($id)
    {
        $data = Input::find($id);

        if (!$data) {
            return response()->json([
                "success" => false,
                "message" => "Input tidak ditemukan"
            ], 404);
        }
        
        // Otorisasi penghapusan
        if ($data->user_id && Auth::id() !== $data->user_id) {
            return response()->json([
                "success" => false,
                "message" => "Akses ditolak atau Input tidak ditemukan"
            ], 403); 
        }

        $data->delete();

        return response()->json([
            "success" => true,
            "message" => "Input berhasil dihapus"
        ]);
    }
}