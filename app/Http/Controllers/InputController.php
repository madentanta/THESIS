<?php

namespace App\Http\Controllers;

use App\Models\Input;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Validator;

class InputController extends Controller
{
    /**
     * STORE INPUT
     * --------------------
     * Endpoint: POST /api/input/store
     */
    public function store(Request $request)
    {
        // VALIDASI
        $validator = Validator::make($request->all(), [
            "soil_ph"       => "required|numeric|between:3,14",
            "location"      => "required|string",
            "temperature"   => "required|numeric",
            "humidity"      => "required|numeric",
            "previous_crop" => "nullable|string",
        ]);

        if ($validator->fails()) {
            return response()->json([
                "success" => false,
                "message" => "Validasi gagal",
                "errors"  => $validator->errors()
            ], 422);
        }

        try {
            // Ambil user_id jika login, kalau tidak → null
            $userId = auth()->id();

            // INSERT DATA
            $input = Input::create([
                "user_id"       => $userId,
                "soil_ph"       => $request->soil_ph,
                "location"      => $request->location,
                "temperature"   => $request->temperature,
                "humidity"      => $request->humidity,
                "previous_crop" => $request->previous_crop,
                "submitted_at"  => now(),
            ]);

            return response()->json([
                "success" => true,
                "message" => "Input berhasil disimpan",
                "data"    => $input
            ], 201);

        } catch (\Throwable $e) {
            return response()->json([
                "success" => false,
                "message" => "Gagal menyimpan data input",
                "error"   => $e->getMessage(),
                "line"    => $e->getLine(),
                "file"    => $e->getFile()
            ], 500);
        }
    }

    /**
     * LIST ALL INPUT
     * --------------------
     * Endpoint: GET /api/input
     */
    public function index()
    {
        try {
            $data = Input::orderBy("input_id", "DESC")->get();

            return response()->json([
                "success" => true,
                "data"    => $data
            ]);
        } catch (\Throwable $e) {
            return response()->json([
                "success" => false,
                "message" => "Gagal mengambil data",
                "error"   => $e->getMessage()
            ], 500);
        }
    }

    /**
     * DETAIL INPUT
     * --------------------
     * Endpoint: GET /api/input/{id}
     */
    public function show($id)
    {
        $data = Input::find($id);

        if (!$data) {
            return response()->json([
                "success" => false,
                "message" => "Input tidak ditemukan"
            ], 404);
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

        $data->delete();

        return response()->json([
            "success" => true,
            "message" => "Input berhasil dihapus"
        ]);
    }
}
