# aibridge-core HTTP server. Stateless: no volumes, no database.
#   docker build -t aibridge-core .
#   docker run -p 8000:8000 -e AIBRIDGE_EXEC_API_TOKENS=change-me aibridge-core
FROM rust:1-slim-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock rustfmt.toml ./
COPY crates ./crates
RUN cargo build --release --locked -p aibridge-core-server --bin aibridge-core

# distroless: no shell, no package manager; runs as uid 65532. TLS roots are compiled in.
FROM gcr.io/distroless/cc-debian12:nonroot
COPY --from=build /src/target/release/aibridge-core /usr/local/bin/aibridge-core
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD ["aibridge-core", "healthcheck", "--port", "8000"]
ENTRYPOINT ["aibridge-core"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
