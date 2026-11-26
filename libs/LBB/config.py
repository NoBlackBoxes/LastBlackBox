# -*- coding: utf-8 -*-
"""
LBB : Config

@author: kampff
"""

# Imports
import json
import pathlib

# Load LBB configuration variables
repo_path = str(pathlib.Path(__file__).resolve().parents[2])
boxes_path = repo_path + "/boxes"
course_path = repo_path + "/course"
libs_path = repo_path + "/libs"
image_prefix = "../../../.."

# Load the single course manifest used by the engine and documentation sites.
course_manifest_path = pathlib.Path(course_path) / "courses.json"
with course_manifest_path.open(encoding="utf8") as manifest_file:
    course_manifest = json.load(manifest_file)["courses"]

required_course_fields = {
    "name", "engine_name", "slug", "description", "logo", "generate", "materials",
    "published"
}
for course in course_manifest:
    missing_fields = required_course_fields - course.keys()
    if missing_fields:
        raise ValueError(
            f"Course {course.get('slug', '<unknown>')} is missing manifest fields: "
            f"{', '.join(sorted(missing_fields))}"
        )
    for flag in ("generate", "materials", "published"):
        if not isinstance(course[flag], bool):
            raise ValueError(f"Course {course['slug']} field {flag} must be true or false")
    if course["published"] and not course["generate"]:
        raise ValueError(f"Published course {course['slug']} must also enable generation")
    if course["materials"] and not course["generate"]:
        raise ValueError(f"Materials course {course['slug']} must also enable generation")

course_names = [course["engine_name"] for course in course_manifest if course["generate"]]
material_course_names = [
    course["engine_name"] for course in course_manifest if course["materials"]
]
course_slugs = {course["engine_name"]: course["slug"] for course in course_manifest}
courses_by_slug = {course["slug"]: course for course in course_manifest}
published_courses = [course for course in course_manifest if course["published"]]

if len(course_slugs) != len(course_manifest):
    raise ValueError("Course engine names must be unique in course/courses.json")
if len({course["slug"] for course in course_manifest}) != len(course_manifest):
    raise ValueError("Course slugs must be unique in course/courses.json")

# Canonical public location used by the site builder and sitemap generator.
site_base_url = "https://noblackboxes.github.io/LastBlackBox"

manifest_slugs = {course["slug"] for course in course_manifest}
version_slugs = {
    path.name for path in (pathlib.Path(course_path) / "versions").iterdir()
    if path.is_dir()
}
if manifest_slugs != version_slugs:
    raise ValueError(
        "Course manifest and course/versions directories differ: "
        f"manifest-only={sorted(manifest_slugs - version_slugs)}, "
        f"directory-only={sorted(version_slugs - manifest_slugs)}"
    )


# Store box names (and order)
box_names = [
    'Atoms',
    'Electrons',
    'Magnets',
    'Light',
    'Sensors',
    'Motors',
    'Transistors',
    'Amplifiers',
    'Circuits',
    'Data',
    'Logic',
    'Memory',
    'FPGAs',
    'Computers',
    'Control',
    'Robotics',
    'Power',
    'Systems',
    'Linux',
    'Python',
    'Networks',
    'Websites',
    'Servers',
    'Security',
    'Audio',
    'Vision',
    'Learning',
    'Intelligence'
]

# Store box names (and order) - Bootcamp
bootcamp_box_names = [
    'Atoms',
    'Electrons',
    'Magnets',
    'Sensors',
    'Motors',
    'Transistors',
    'Data',
    'Logic',
    'Memory',
    'Computers',
    'Control',
    'Robotics',
    'Power',
    'Systems',
    'Linux',
    'Python',
    'Networks',
    'Websites',
    'Servers',
    'Audio',
    'Vision'
]

# Store box names (and order) - Braitenberg
braitenberg_box_names = [
    'Atoms',
    'Electrons',
    'Magnets',
    'Sensors',
    'Motors',
    'Transistors',
    'Data',
    'Logic',
    'Memory',
    'Computers',
    'Control',
    'Robotics'
]

# Store box names (and order) - Build a Brain
buildabrain_box_names = [
    'Sensors',
    'Motors',
    'Transistors',
    'Computers',
    'Robotics',
    'Systems',
    'Networks',
    'Intelligence'
]

# Store box names (and order) - Own Phone
ownphone_box_names = [
    'Electrons',
    'Magnets',
    'Light',
    'Transistors',
    'Data',
    'Computers',
    'Systems',
    'Networks',
    'Security'
]

# Specify package box names (7 x 7 grid)

# Large (6 x 4)
large_box_names = [
    'Atoms',
    'Electrons',
    'Magnets',
    'Light',
    'Amplifiers',
    'Circuits',
    'Data',
    'Logic',
    'Control',
    'Robotics',
    'Power',
    'Systems',
    'Websites',
    'Servers',
    'Security',
    'Audio',
    'Atoms',
    'Electrons',
    'Magnets',
    'Light',
    'Amplifiers',
    'Circuits',
    'Data',
    'Logic'
]

# Medium (4 x 3)
medium_box_names = [
    'Sensors',
    'Motors',
    'Transistors',
    'Memory',
    'FPGAs',
    'Computers',
    'Linux',
    'Python',
    'Networks',
    'Vision',
    'Learning',
    'Intelligence'
]

# Small (2 x 3)
small_box_names = [
    'Sensors',
    'Motors',
    'Transistors',
    'Memory',
    'FPGAs',
    'Computers',
    'Linux',
    'Python',
    'Networks'
]

# Cables (1 x 7)
cables_box_names = [
    'Websites',
    'Servers',
    'Security',
    'Audio',
    'Vision',
    'Learning',
    'Intelligence',
    'Atoms',
    'Electrons',
    'Magnets',
    'Light',
    'Sensors',
    'Motors',
    'Transistors',
    'Amplifiers',
    'Circuits',
    'Data',
    'Logic',
    'Memory',
    'FPGAs',
    'Computers'
]

#FIN
