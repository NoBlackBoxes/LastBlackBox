import os
import subprocess
import markdown

# Get LBB root
LBBROOT = os.environ.get('LBBROOT')

# Set site root
site_path = LBBROOT + '/repo/site/lastblackbox.training'

# Create site root (if not present)
ret = subprocess.call(["mkdir", "-p", site_path])

# Set html folder
html_path = LBBROOT + '/repo/site/html'

# Set css folder
css_path = LBBROOT + '/repo/site/css'

# Sync images to site
rsync_call = []
rsync_call.append("rsync")
rsync_call.append("-am")
rsync_call.append("--exclude=.vscode/")
rsync_call.append("--exclude=admin/")
rsync_call.append("--exclude=site/")
rsync_call.append("--include=*/")
rsync_call.append("--include=*.png")
rsync_call.append("--exclude=*")
rsync_call.append(LBBROOT + "/repo/")
rsync_call.append(site_path)
rsync_call.append("--delete")
ret = subprocess.call(rsync_call)

# Copy index.html
ret = subprocess.call(["cp", html_path + "/index.html", site_path + "/"])

# Copy style.css
ret = subprocess.call(["cp", css_path + "/style.css", site_path + "/"])

# Sync html pages to site

# Load header
header_path = html_path + '/header.html'
file = open(header_path,"r")
header_text = file.read()
file.close()

# Load footer
footer_path = html_path + '/footer.html'
file = open(footer_path,"r")
footer_text = file.read()
file.close()

# Set *.md path
input_path = LBBROOT + '/repo/README.md'

# Set *.html path
output_path = site_path + '/README.html'

# Load input markdown
file = open(input_path,"r")
text = file.read()
file.close()

# Parse markdown
parsed = markdown.markdown(text)

# Save output html
file = open(output_path,"w")
num_written = file.write(header_text)
num_written = file.write(parsed)
num_written = file.write(footer_text)
file.close()

#FIN