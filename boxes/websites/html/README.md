# Websites : HTML

Hypertext Markup Language

## Basic
A website is just a text file with special tags, e.g. \<TAG>, that control how the text (and other content) is displayed by a web browser.

```html
<!DOCTYPE html>
<html>

<body>
This is a <b>website</b>
</body>

</html>
```

Create the above text file and open it with a web browser.

We can also divide and format the page as we see fit.

```html
<!DOCTYPE html>
<html>
<body>
    <div class=viewport style="display: flex; width: 100%; justify-content: center">
        <div class=page style="width: 80%;">
            <div class=title style="text-align: center; background: lightblue">
                <h1>This is the Title</h1>
                <hr>
            </div>
            <div class=container>
                <div class=sidebar style="background: yellow; float: left; width: 25%; height: 300px;">
                    This is the sidebar:
                    <ul>
                        <li>item1
                        <li>item2
                        <li>item3
                    </ul>
                </div>
                <div class=content style="background: lime; float: left; width: 75%; height: 300px;">
                    This is the content
                </div>
            </div>
        </div>
    </div>
</body>
</html>
```
