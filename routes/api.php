<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\AuthController;
use App\Http\Controllers\InputController;
use App\Http\Controllers\DatasetController;
use App\Http\Controllers\ResultController;


// ---------------------------------------------------------
// 1. RUTE PUBLIK & GUEST (Tidak memerlukan Token/Auth Apapun)
// ---------------------------------------------------------

// ==== AUTH (PUBLIC) ====
Route::post("/register", [AuthController::class, "register"]);
Route::post("/login", [AuthController::class, "login"]);
Route::post("/forgot-password", [AuthController::class, "forgotPassword"]);
Route::post("/reset-password", [AuthController::class, "resetPassword"]);

// PENAMBAHAN FUNGSI VERIFIKASI EMAIL
Route::get("/verify-email", [AuthController::class, "verifyEmail"]); 

// ==== INPUT & REKOMENDASI (GUEST / SERVICE) ====
Route::post("/input/store", [InputController::class, "store"]);
Route::get("/recommendation/{input_id}", [ResultController::class, "getRecommendation"]);


// ---------------------------------------------------------
// 2. USER (TOKEN: auth:sanctum)
// ---------------------------------------------------------
Route::middleware("auth:sanctum")->group(function () {
    
    // ==== AUTH (USER) ====
    Route::post("/logout", [AuthController::class, "logout"]);

    // MENDAPATKAN STATUS VERIFIKASI PENGGUNA YANG SEDANG LOGIN
    Route::get("/user/status", [AuthController::class, "getUserStatus"]); 

    // ==== INPUT (USER DATA) ====
    Route::get("/input/me", [InputController::class, "index"]); 
    Route::get("/input/me/{id}", [InputController::class, "show"]); 

});


// ---------------------------------------------------------
// 3. ADMIN (ADMIN ONLY: basic.admin)
// ---------------------------------------------------------
Route::middleware("basic.admin")->group(function () {

    // === DATASET ADMIN ===
    Route::post("/dataset/upload", [DatasetController::class, "upload"]);
    Route::get("/dataset/list", [DatasetController::class, "list"]);
    Route::post("/dataset/list-filtered", [DatasetController::class, "listFiltered"]);
    Route::delete("/dataset/delete", [DatasetController::class, "deleteAll"]);
    Route::delete("/dataset/delete/{id}", [DatasetController::class, "deleteById"]);

    // === INPUT ADMIN (Manajemen Data Input Global) ===
    Route::get("/admin/input", [InputController::class, "adminIndex"]);
    Route::get("/admin/input/{id}", [InputController::class, "adminShow"]);
    Route::delete("/input/delete/{id}", [InputController::class, "delete"]);
});