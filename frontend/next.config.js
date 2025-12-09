/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    domains: [],
  },
  // Enable standalone output for Docker deployment
  output: 'standalone',
}

module.exports = nextConfig
