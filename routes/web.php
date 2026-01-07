<?php
use App\Http\Controllers\ResultController;
use Illuminate\Support\Facades\Route;

Route::get('/', function () {
    return redirect('beranda.html');
});

// Route ini yang akan dipanggil oleh JavaScript di beranda.html
// routes/web.php
Route::get('/predict-ai/{input_id}', [ResultController::class, 'getRecommendation']);