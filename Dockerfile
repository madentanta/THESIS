# Base image
FROM php:8.2-fpm

# Install dependencies sistem
RUN apt-get update && apt-get install -y \
    git \
    unzip \
    libzip-dev \
    libpq-dev \
    && docker-php-ext-install pdo pdo_mysql pdo_pgsql pgsql zip

# Set working directory sesuai docker-compose volumes
WORKDIR /var/www/app

# Copy project files
COPY . .

# Pastikan folder storage dan cache bisa ditulis
RUN chown -R www-data:www-data /var/www/app/storage /var/www/app/bootstrap/cache

# Install composer
RUN php -r "copy('https://getcomposer.org/installer', 'composer-setup.php');" \
    && php composer-setup.php --install-dir=/usr/local/bin --filename=composer \
    && php -r "unlink('composer-setup.php');"

# Install PHP dependencies
RUN composer install --no-dev --optimize-autoloader --no-scripts

# Expose port PHP-FPM internal
EXPOSE 9000

# Start PHP-FPM
CMD ["php-fpm"]
