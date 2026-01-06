FROM php:8.3-fpm

RUN apt-get update && apt-get install -y \
    git zip unzip libpq-dev \
    && docker-php-ext-install pdo pdo_pgsql

WORKDIR /var/www/app
