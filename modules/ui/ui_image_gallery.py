import gradio as gr
from shared import path_manager, translate
from modules.imagebrowser import ImageBrowser

browser = ImageBrowser()
t = translate

def create_image_gallery():
    with gr.Blocks(theme=gr.themes.Soft()) as app_image_browser:
        with gr.Row():
            # Left side for gallery
            with gr.Column(scale=2):
                gallery = gr.Gallery(
                    label=t("Images"),
                    show_label=False,
                    columns=[4],
                    height="600px",
                    object_fit="contain",
                    value=browser.load_images(1)[0],
                )
                ib_page = gr.Slider(
                    label=t("Page"),
                    value=1,
                    step=1,
                    minimum=1,
                    maximum=browser.num_images_pages()[1],
                )
                ib_range = gr.Markdown()

            # Right side for metadata and search
            with gr.Column(scale=1):
                with gr.Row():
                    update_btn = gr.Button(t("Update DB"), scale=5)
                    gr.HTML(value="""<a href="gradio_api/file/html/slideshow.html" style="color: gray; text-decoration: none" target="_blank">🛝</a>""")
                metadata_output = gr.Textbox(
                    label=t("Image Metadata"), interactive=False, lines=15
                )
                search_input = gr.Textbox(
                    label=t("Search Metadata"), placeholder=t("Search term")
                )
                search_btn = gr.Button(t("Search"))
                status_output = gr.Markdown()

        # Event handlers
        update_btn.click(
            fn=browser.update_images,
            api_visibility='undocumented',
            outputs=[gallery, ib_page, status_output],
        )
        ib_page.change(
            fn=browser.load_images,
            api_visibility='undocumented',
            inputs=[ib_page],
            outputs=[gallery, ib_range],
        )
        gallery.select(
            fn=browser.get_image_metadata,
            api_visibility='undocumented',
            outputs=[metadata_output],
        )
        search_btn.click(
            fn=browser.search_metadata,
            api_visibility='undocumented',
            inputs=[search_input],
            outputs=[gallery, ib_page, status_output],
        )
        search_input.submit(
            fn=browser.search_metadata,
            api_visibility='undocumented',
            inputs=[search_input],
            outputs=[gallery, ib_page, status_output],
        )


    return app_image_browser


# Optional: If you want to launch just this gallery
def launch_image_gallery():
    app_image_browser = create_image_gallery()
    app_image_browser.launch()
