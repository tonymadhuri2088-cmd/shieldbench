"""Wraps site_src/fragment.html into docs/index.html (a full page for GitHub Pages and the local server)."""
frag = open("site_src/fragment.html", encoding="utf-8").read()
head, rest = frag.split('<header class="top">', 1)
open("docs/index.html", "w", encoding="utf-8").write(
    '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
    + head + '</head>\n<body>\n<header class="top">' + rest + '\n</body>\n</html>\n')
print("wrote docs/index.html")
