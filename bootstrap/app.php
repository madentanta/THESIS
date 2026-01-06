<?php

use Illuminate\Foundation\Application;
use Illuminate\Foundation\Configuration\Exceptions;
use Illuminate\Foundation\Configuration\Middleware;
use Illuminate\Http\Request;
use Illuminate\Validation\ValidationException;
use Symfony\Component\HttpKernel\Exception\BadRequestHttpException;

return Application::configure(basePath: dirname(__DIR__))
    ->withRouting(
        web: __DIR__ . '/../routes/web.php',
        api: __DIR__ . '/../routes/api.php',
        commands: __DIR__ . '/../routes/console.php',
        health: '/up',
    )
    ->withMiddleware(function (Middleware $middleware) {

        // Route Middleware Aliases
        $middleware->alias([
            "auth"         => \App\Http\Middleware\Authenticate::class,
            "auth.token"   => \App\Http\Middleware\AuthTokenMiddleware::class,
            "guest"        => \App\Http\Middleware\RedirectIfAuthenticated::class,
            "basic.admin"  => \App\Http\Middleware\BasicAuthAdmin::class,
        ]);

    })
    ->withExceptions(function (Exceptions $exceptions) {

        /**
         * 1. JSON Syntax Error (400)
         */
        $exceptions->render(function (BadRequestHttpException $e, Request $request) {
            if ($request->is('api/*')) {

                $msg = strtolower($e->getMessage());

                if (str_contains($msg, 'malformed') ||
                    str_contains($msg, 'syntax') ||
                    str_contains($msg, 'json')) {

                    return response()->json([
                        'status'  => 'error',
                        'message' => 'Format Request Tidak Valid (Bad Request).',
                        'detail'  => 'Kesalahan sintaks JSON pada body request.',
                    ], 400);
                }
            }

            return null;
        });


        /**
         * 2. ValidationException (422)
         * – FIX untuk FE kamu
         */
        $exceptions->render(function (ValidationException $e, Request $request) {
            if ($request->is('api/*')) {
                return response()->json([
                    'message' => 'The given data was invalid.',
                    'errors'  => $e->errors(),
                ], 422);
            }

            return null;
        });

    })
    ->create();
