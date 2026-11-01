"""Wraps site_src/fragment.html into docs/index.html (a full page for GitHub Pages and the local server)."""
AUTHOR = "Tony Madhuri"
COLLEGE = "IIIT Raichur"
import os
LOGO = next((f for f in ("college-logo.png","college-logo.jpg","college-logo.svg","college-logo.webp") if os.path.exists("docs/"+f)), None)
frag = open("site_src/fragment.html", encoding="utf-8").read()
head, rest = frag.split('<header class="top">', 1)
def fill(t):
    t = t.replace("{{AUTHOR}}", AUTHOR).replace("{{COLLEGE}}", COLLEGE).replace("{{INITIAL}}", AUTHOR[:1].upper())
    logo = ('<img class="clogo" src="%s" alt="%s logo">' % (LOGO, COLLEGE)) if LOGO else ""
    return t.replace("{{LOGO}}", logo)
open("docs/index.html", "w", encoding="utf-8").write(fill(
    '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
    + head + '</head>\n<body>\n<header class="top">' + rest + '\n</body>\n</html>\n'))
print("wrote docs/index.html")
