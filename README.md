# 🤖 Bitcoin Trading Bot — HTTPS Setup

## Architektur

```
Internet
   │
   ▼
[Nginx :443]  ← TLS, Basic Auth
   │
   ▼
[Flask :5000] ← Dashboard (intern)
   │
[charts_data] ← Docker Volume
   │
[Trading Bot] ← Signale & Charts

[DuckDNS]  → IP-Update alle 5 Min
[Certbot]  → Zertifikat-Erneuerung alle 12h
```

---

## Schritt 1 — DuckDNS Subdomain (5 Min)

1. **https://www.duckdns.org** aufrufen & einloggen
2. Subdomain wählen z.B. `meinbot` → `meinbot.duckdns.org`
3. **Token** kopieren (steht oben auf der Seite)

---

## Schritt 2 — Konfiguration

```bash
cp .env.example .env
nano .env
```

Pflichtfelder:
```ini
DUCKDNS_DOMAIN=meinbot            # nur Subdomain-Teil ohne .duckdns.org
DUCKDNS_TOKEN=xxxxxxxx-xxxx-...   # von duckdns.org
LETSENCRYPT_EMAIL=du@mail.de
DASHBOARD_USER=admin
DASHBOARD_PASS=sicheres-passwort
```

---

## Schritt 3 — Hetzner Firewall

In der Hetzner Cloud Console → Firewall → Eingehend:

| Port | Protokoll |
|------|-----------|
| 80   | TCP       |
| 443  | TCP       |

---

## Schritt 4 — Ersteinrichtung (einmalig)

```bash
chmod +x init-letsencrypt.sh

# Erst im Staging testen (kein Rate-Limit bei Let's Encrypt)
# STAGING=1 in .env setzen, dann:
./init-letsencrypt.sh

# Wenn erfolgreich: STAGING=0 setzen und wiederholen
./init-letsencrypt.sh
```

---

## Schritt 5 — Normalbetrieb

```bash
docker-compose up -d
```

→ **https://meinbot.duckdns.org**

---

## Zertifikat erneuern (automatisch)

Certbot prüft alle 12h — keine manuelle Aktion nötig.

Manuell:
```bash
docker-compose run --rm certbot renew
docker-compose exec nginx-proxy nginx -s reload
```

## Passwort ändern

```bash
docker run --rm -v "$(pwd)/nginx:/etc/nginx" \
    httpd:alpine htpasswd -Bbc /etc/nginx/.htpasswd USER PASSWORT
docker-compose exec nginx-proxy nginx -s reload
```

## Logs

```bash
docker-compose logs -f trading-bot   # Bot
docker-compose logs -f nginx-proxy   # Nginx
docker-compose logs -f duckdns       # IP-Updates
docker-compose logs -f certbot       # Zertifikat
```

## Troubleshooting

**Zertifikat schlägt fehl** → Port 80 in Hetzner Firewall offen? DuckDNS-Token korrekt?

**IP hat sich geändert** → DuckDNS-Container aktualisiert automatisch alle 5 Min.
Manuell: `docker-compose restart duckdns`
