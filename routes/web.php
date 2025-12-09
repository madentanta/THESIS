<?php

use Illuminate\Support\Facades\Route;

// Mengarahkan root URL (/) langsung ke file beranda.html
Route::get('/', function () {
    // Redirect klien (browser) dari "/" ke "/beranda.html"
    return redirect('beranda.html');
});