# Undertale: Inference Frontend

Undertale inference frontend.

## Description

A lightweight LLM-like inference UI that sits on top of the Undertale inference
server API.

## Installation

### Prerequisites

- [NGINX][nginx] as a reverse proxy (to serve static files and proxy `/api/`)
- The inference server must be installed and running

[nginx]: https://nginx.org/

### Installing

Copy the production bundle to the NGINX web root:

```bash
cp -r dist/undertale-frontend/browser/* /var/www/html/
```

Configure NGINX to serve the static files using the example configuration as a
reference:

```bash
cp examples/nginx.conf /etc/nginx/conf.d/undertale-frontend.conf
nginx -s reload
```

## Usage

Access the app via a web browser.

## Contributing

### Prerequisites

The main Undertale conda environment and the inference server must be set up
before developing the inference frontend. See the `Installation` section of the
documentation and the inference server README for more details.

### Development Environment

Update the existing `undertale` conda environment with the inference frontend's
development dependencies:

```bash
conda env update -f environment.development.yml
conda activate undertale
```

### Development Server

```bash
npm start
```

This runs `ng serve` and starts the Angular dev server at
`http://localhost:4200`. It proxies all `/api/` requests to
`http://localhost:5000` (the Flask dev server). The inference server must be
running before starting the development frontend.

### Production Bundle

To deploy the app, build the production bundle:

```bash
npm run build
```
