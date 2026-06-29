# Resources

## Icon

`icon.png` is the source image for the application icon. It is **not** shipped
with the package; only the generated `.ico` is.

To regenerate the `.ico` after changing `icon.png`:

```
uv run python -c "from PIL import Image; Image.open('icon.png').save('../src/body_eye_sync/resources/icon.ico', sizes=[(16,16),(24,24),(32,32),(48,48),(64,64),(128,128),(256,256)])"
```
