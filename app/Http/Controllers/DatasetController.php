<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;
use PhpOffice\PhpSpreadsheet\IOFactory;
use Symfony\Component\HttpFoundation\Response;

class DatasetController extends Controller
{
    /**
     * UPLOAD DATASET (APPEND MODE)
     */
public function upload(Request $req)
{
    $req->validate([
        "file" => "required|mimes:xlsx,xls"
    ]);

    try {
        $file = $req->file("file");
        $fileName = time() . '_' . $file->getClientOriginalName();
        
        // Simpan ke folder tmp secara manual agar path-nya absolut
        $destinationPath = storage_path("app/tmp");
        
        // Pastikan folder ada
        if (!file_exists($destinationPath)) {
            mkdir($destinationPath, 0777, true);
        }

        $file->move($destinationPath, $fileName);
        $fullPath = $destinationPath . '/' . $fileName;

        // Cek apakah file benar-benar ada dan bisa dibaca
        if (!file_exists($fullPath) || !is_readable($fullPath)) {
            throw new \Exception("File tidak ditemukan atau tidak dapat dibaca di: " . $fullPath);
        }

        // Load Excel
        $spreadsheet = IOFactory::load($fullPath);
        $sheet = $spreadsheet->getActiveSheet()->toArray();

        $buffer   = [];
        $inserted = 0;
        $skipped  = 0;

        DB::beginTransaction();

        foreach ($sheet as $i => $row) {
            if ($i === 0) continue; // Skip header

            // Validasi kolom minimal
            if (empty($row[0]) || empty($row[7]) || empty($row[8])) {
                $skipped++;
                continue;
            }

            $buffer[] = [
                "nama_daerah"  => trim($row[0]),
                "fertility"    => (float) ($row[1] ?? 0),
                "moisture"     => (float) ($row[2] ?? 0),
                "ph"           => (float) ($row[3] ?? 0),
                "temp"         => (float) ($row[4] ?? 0),
                "sunlight"     => (float) ($row[5] ?? 0),
                "humidity"     => (float) ($row[6] ?? 0),
                "kecamatan"    => trim($row[7]),
                "nama_tanaman" => trim($row[8]),
                "created_at"   => now()
            ];

            if (count($buffer) >= 200) {
                DB::table("dataset_tanaman")->insert($buffer);
                $inserted += count($buffer);
                $buffer = [];
            }
        }

        if (!empty($buffer)) {
            DB::table("dataset_tanaman")->insert($buffer);
            $inserted += count($buffer);
        }

        DB::commit();
        unlink($fullPath); // Hapus file sementara

        return response()->json([
            "status"   => "success",
            "message"  => "Dataset berhasil di-upload",
            "inserted" => $inserted,
            "skipped"  => $skipped
        ], Response::HTTP_OK);

    } catch (\Exception $e) {
        DB::rollBack();
        if (isset($fullPath) && file_exists($fullPath)) unlink($fullPath);

        return response()->json([
            "status"  => "error",
            "message" => "Gagal upload: " . $e->getMessage()
        ], 500);
    }
}



    /**
     * LIST SEMUA DATASET
     */
    public function list()
    {
        $datasets = DB::table("dataset_tanaman")
            ->orderBy("id", "desc")
            ->get();

        $total = $datasets->count();

        $message = ($total > 0)
            ? "Data dataset berhasil diambil."
            : "Dataset kosong. Silakan upload data baru.";

        return response()->json([
            'status'        => 'success',
            'message'       => $message,
            'total_records' => $total,
            'data'          => $datasets
        ], Response::HTTP_OK);
    }


    /**
     * LIST FILTERED (Kecamatan + Nama Tanaman)
     */
    public function listFiltered(Request $req)
    {
        try {
            // RULE VALIDASI
            $rules = [
                'kecamatan_filter'   => 'sometimes|array|min:1',
                'kecamatan_filter.*' => 'required|string|min:1',

                'nama_tanaman_filter'   => 'sometimes|array|min:1',
                'nama_tanaman_filter.*' => 'required|string|min:1',

                'limit' => 'integer|min:1|max:100',
            ];

            // CUSTOM MESSAGE
            $messages = [
                'kecamatan_filter.array'     => 'The filter for kecamatan must be a list (array).',
                'kecamatan_filter.min'       => 'kecamatan_filter must contain at least one value.',
                'kecamatan_filter.*.required'=> 'Each item in kecamatan_filter must not be empty.',
                'kecamatan_filter.*.string'  => 'Each item in kecamatan_filter must be a string.',
                'kecamatan_filter.*.min'     => 'Each item in kecamatan_filter must not be empty.',

                'nama_tanaman_filter.array'     => 'The filter for nama_tanaman must be a list (array).',
                'nama_tanaman_filter.min'       => 'nama_tanaman_filter must contain at least one value.',
                'nama_tanaman_filter.*.required'=> 'Each item in nama_tanaman_filter must not be empty.',
                'nama_tanaman_filter.*.string'  => 'Each item in nama_tanaman_filter must be a string.',
                'nama_tanaman_filter.*.min'     => 'Each item in nama_tanaman_filter must not be empty.',

                'limit.integer' => 'Limit must be an integer.',
            ];

            $req->validate($rules, $messages);

            // Minimal salah satu filter harus ada
            if (
                !$req->has('kecamatan_filter') &&
                !$req->has('nama_tanaman_filter')
            ) {
                return response()->json([
                    'status'  => 'error',
                    'message' => 'Mandatory parameter cannot be null or empty. At least one filter is required.',
                ], 400);
            }

            // AMBIL INPUT
            $kecamatanFilters = $req->input('kecamatan_filter', []);
            $tanamanFilters   = $req->input('nama_tanaman_filter', []);
            $limit            = $req->input('limit', 15);

            // QUERY
            $query = DB::table('dataset_tanaman');

            if (!empty($kecamatanFilters)) {
                $query->whereIn('kecamatan', $kecamatanFilters);
            }
            if (!empty($tanamanFilters)) {
                $query->whereIn('nama_tanaman', $tanamanFilters);
            }

            $datasets = $query->orderBy('id', 'desc')->paginate($limit);

            $totalRecords = $datasets->total();
            $responseMessage = $totalRecords > 0
                ? 'Data dataset berhasil diambil.'
                : 'Dataset Tidak Ditemukan!';

            return response()->json([
                'status'        => 'success',
                'message'       => $responseMessage,
                'total_records' => $totalRecords,
                'data' => [
                    'records'    => $datasets->items(),
                    'pagination' => [
                        'total'        => $totalRecords,
                        'per_page'     => $datasets->perPage(),
                        'current_page' => $datasets->currentPage(),
                        'last_page'    => max($datasets->lastPage(), 1),
                        'next_page'    => $datasets->hasMorePages() ? $datasets->currentPage() + 1 : null,
                        'prev_page'    => $datasets->previousPageUrl() ? $datasets->currentPage() - 1 : null,
                    ]
                ]
            ], 200);

        } catch (\Illuminate\Validation\ValidationException $e) {

            return response()->json([
                'status'  => 'error',
                'message' => 'Payload request tidak sesuai.',
                'errors'  => $e->errors()
            ], 400);

        } catch (\Exception $e) {

            return response()->json([
                'status'  => 'error',
                'message' => 'Bad Request',
                'detail'  => 'Terdapat kesalahan pada body request.'
            ], 400);
        }
    }


    /**
     * DELETE BY ID
     */
    public function deleteById($id)
    {
        $dataset = DB::table("dataset_tanaman")->where("id", $id)->first();

        if (!$dataset) {
            return response()->json([
                "error" => "Dataset dengan ID $id tidak ditemukan"
            ], Response::HTTP_NOT_FOUND);
        }

        DB::table("dataset_tanaman")->where("id", $id)->delete();

        return response()->json([
            "message" => "Dataset berhasil dihapus",
            "deleted_info" => [
                "id"           => $id,
                "nama_tanaman" => $dataset->nama_tanaman,
                "kecamatan"    => $dataset->kecamatan
            ]
        ], Response::HTTP_OK);
    }


    /**
     * DELETE ALL DATASET
     */
    public function deleteAll()
    {
        $count = DB::table("dataset_tanaman")->count();

        if ($count === 0) {
            return response()->json([
                "status"  => "error",
                "message" => "Dataset kosong. Tidak ada data yang dapat dihapus."
            ], Response::HTTP_BAD_REQUEST);
        }

        DB::table("dataset_tanaman")->truncate();

        return response()->json([
            "status"          => "success",
            "message"         => "Semua dataset berhasil dihapus.",
            "deleted_records" => $count
        ], Response::HTTP_OK);
    }
}