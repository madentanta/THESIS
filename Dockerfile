# Base image
FROM php:8.2-fpm

# Install system dependencies + PHP extensions
RUN apt-get update && apt-get install -y \
    git \
    unzip \
    libzip-dev \
    libpq-dev \
    libicu-dev \
    && docker-php-ext-install \
        pdo \
        pdo_mysql \
        pdo_pgsql \
        pgsql \
        zip \
        intl

# Set working directory
WORKDIR /var/www/app

# Copy project files
COPY . .

# Install composer
RUN php -r "copy('https://getcomposer.org/installer', 'composer-setup.php');" \
    && php composer-setup.php --install-dir=/usr/local/bin --filename=composer \
    && php -r "unlink('composer-setup.php');"

# Install PHP dependencies
RUN composer install --no-dev --optimize-autoloader --no-scripts

# Permission for Laravel
RUN chown -R www-data:www-data /var/www/app/storage /var/www/app/bootstrap/cache

# Expose port PHP-FPM
EXPOSE 9000

# Start PHP-FPM
CMD ["php-fpm"]
