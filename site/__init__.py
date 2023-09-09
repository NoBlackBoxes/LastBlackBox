from flask import Flask, redirect, render_template, request, url_for
import markdown
import markdown.extensions.fenced_code
import markdown.extensions.codehilite
from pygments.formatters import HtmlFormatter

# Create Flask app
app = Flask(__name__)

# Specify root route
@app.route("/", methods=("GET", "POST"))
def index():
    readme_file = open("templates/README.md", "r")
    print(readme_file)
    md_template_string = markdown.markdown(readme_file.read(), extensions=["fenced_code", "codehilite"])
    
    # Generate CSS for syntax highlighting
    formatter = HtmlFormatter(style="emacs",full=True,cssclass="codehilite")
    css_string = formatter.get_style_defs()
    md_css_string = "<style>" + css_string + "</style>"
    print(css_string)

    md_template = md_css_string + md_template_string
    return md_template
    #return render_template("index.html")

# MAIN
if __name__ == "__main__":
    app.run()

#FIN