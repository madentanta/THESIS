<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Mail;
use Carbon\Carbon;

class SendAiTrendReport extends Command
{
    protected $signature = 'report:ai-trend';
    protected $description = 'Send Plant Advisor Weekly Report';

    public function handle()
    {
        $today = Carbon::now();
        $start = $today->copy()->subDays(7);

        /*
        |--------------------------------------------------------------------------
        | 1. Total Requests
        |--------------------------------------------------------------------------
        */
        $totalRequest = DB::table('input')
            ->whereBetween('submitted_at', [$start, $today])
            ->count();

        /*
        |--------------------------------------------------------------------------
        | 2. Average Confidence
        |--------------------------------------------------------------------------
        */
        $avgConfidence = DB::table('crop_recommendation')
            ->whereBetween('recommended_at', [$start, $today])
            ->avg('score');

        $avgConfidence = $avgConfidence
            ? round($avgConfidence * 100, 2)
            : 0;

        /*
        |--------------------------------------------------------------------------
        | 3. Top Recommendations
        |--------------------------------------------------------------------------
        */
        $topRekomendasi = DB::table('crop_recommendation')
            ->whereBetween('recommended_at', [$start, $today])
            ->select('recommended_crop', DB::raw('COUNT(*) as total'))
            ->groupBy('recommended_crop')
            ->orderByDesc('total')
            ->limit(5)
            ->get();

        /*
        |--------------------------------------------------------------------------
        | 4. Top Locations (Case-Insensitive + Trim)
        |--------------------------------------------------------------------------
        */
        $topLokasi = DB::table('input')
            ->whereBetween('submitted_at', [$start, $today])
            ->selectRaw('LOWER(TRIM(location)) as location, COUNT(*) as total')
            ->groupBy(DB::raw('LOWER(TRIM(location))'))
            ->orderByDesc('total')
            ->limit(5)
            ->get();

        /*
        |--------------------------------------------------------------------------
        | 5. Daily Trend (Last 7 Days)
        |--------------------------------------------------------------------------
        */
        $daily = DB::table('input')
            ->whereBetween('submitted_at', [$start, $today])
            ->selectRaw("DATE(submitted_at) as date, COUNT(*) as total")
            ->groupBy('date')
            ->orderBy('date')
            ->get()
            ->keyBy('date');

        $dates = [];
        $totals = [];

        for ($i = 6; $i >= 0; $i--) {
            $date = Carbon::now()->subDays($i)->format('Y-m-d');
            $dates[] = Carbon::parse($date)->format('d M');
            $totals[] = $daily[$date]->total ?? 0;
        }

        /*
        |--------------------------------------------------------------------------
        | Generate Professional Line Chart PNG
        |--------------------------------------------------------------------------
        */

        $width = 1000;
        $height = 500;

        $image = imagecreate($width, $height);

        // Colors
        $white = imagecolorallocate($image, 255, 255, 255);
        $black = imagecolorallocate($image, 0, 0, 0);
        $blue  = imagecolorallocate($image, 0, 102, 204);
        $gray  = imagecolorallocate($image, 220, 220, 220);

        imagefill($image, 0, 0, $white);

        // Chart margins
        $marginLeft = 100;
        $marginRight = 60;
        $marginTop = 80;
        $marginBottom = 80;

        $chartWidth = $width - $marginLeft - $marginRight;
        $chartHeight = $height - $marginTop - $marginBottom;

        // Chart border
        imagerectangle(
            $image,
            $marginLeft,
            $marginTop,
            $marginLeft + $chartWidth,
            $marginTop + $chartHeight,
            $black
        );

        // Dynamic scaling
        $maxValue = max($totals);
        $maxValue = $maxValue > 0 ? ceil($maxValue * 1.2) : 1;

        // Grid lines + Y-axis labels
        $gridLines = 5;
        for ($i = 0; $i <= $gridLines; $i++) {

            $y = $marginTop + ($chartHeight / $gridLines) * $i;

            imageline(
                $image,
                $marginLeft,
                $y,
                $marginLeft + $chartWidth,
                $y,
                $gray
            );

            $value = round($maxValue - ($maxValue / $gridLines) * $i);
            imagestring($image, 3, 40, $y - 7, $value, $black);
        }

        // Plot points
        $count = count($totals);
        $pointSpacing = $count > 1 ? $chartWidth / ($count - 1) : 0;

        $points = [];

        foreach ($totals as $index => $value) {

            $x = $marginLeft + ($index * $pointSpacing);
            $y = $marginTop + $chartHeight - (($value / $maxValue) * $chartHeight);

            $points[] = [$x, $y];

            imagefilledellipse($image, $x, $y, 10, 10, $blue);

            // Date label
            imagestring(
                $image,
                3,
                $x - 20,
                $marginTop + $chartHeight + 10,
                $dates[$index],
                $black
            );
        }

        // Connect lines
        for ($i = 0; $i < count($points) - 1; $i++) {
            imageline(
                $image,
                $points[$i][0],
                $points[$i][1],
                $points[$i + 1][0],
                $points[$i + 1][1],
                $blue
            );
        }

        // Title
        $title = "Plant Advisor Request Trend (Last 7 Days)";
        imagestring(
            $image,
            5,
            ($width / 2) - (strlen($title) * 4),
            30,
            $title,
            $black
        );

        $chartPath = storage_path('app/public/ai_trend.png');
        imagepng($image, $chartPath);
        imagedestroy($image);

        /*
        |--------------------------------------------------------------------------
        | Email Content
        |--------------------------------------------------------------------------
        */

        $content = "
PLANT ADVISOR WEEKLY REPORT

Total Requests      : $totalRequest
Average Confidence  : $avgConfidence %

Top Recommendations:
";

        foreach ($topRekomendasi as $row) {
            $content .= "- {$row->recommended_crop} ({$row->total})\n";
        }

        $content .= "\nTop Locations:\n";

        foreach ($topLokasi as $row) {
            $location = ucfirst($row->location);
            $content .= "- {$location} ({$row->total})\n";
        }

        /*
        |--------------------------------------------------------------------------
        | Send Email
        |--------------------------------------------------------------------------
        */

        Mail::raw($content, function ($message) use ($chartPath) {
            $message->to(env('REPORT_EMAIL'))
                    ->subject('Plant Advisor Weekly Report')
                    ->attach($chartPath);
        });

        $this->info('Plant Advisor Weekly Report Sent Successfully!');
    }
}
