from typing import List, Dict, Any, Tuple
import tempfile, os, uuid, base64
import pathlib
import pandas as pd
import datetime
import inspect
import json
import io  # <-- Required for figure handling

# Define the placeholder you will use in your add-ons
ATTACHMENT_PLACEHOLDER = "[view_details_link]"
try:
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    Figure = type(None) 
    HAS_MATPLOTLIB = False

try:
    from plotly.graph_objects import Figure as PlotlyFigure
    HAS_PLOTLY = True
except ImportError:
    PlotlyFigure = type(None)
    HAS_PLOTLY = False

# Define the placeholder you will use in your add-ons
ATTACHMENT_PLACEHOLDER = "[view_details_link]"
class BaseAddOn:
    

    # ------------------- Data Processing Stages ------------------- #
    # (before_sequence, apply_window, transformation, etc. are unchanged)
    def before_sequence(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]: return state
    def apply_window(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]: return state
    def transformation(self, state: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]: return state
    def on_evaluation(self, eval_data: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]: return eval_data
    def on_evaluation_end(self, eval_data: Dict[str, Any], pipeline_extra_info: Dict[str, Any]) -> Dict[str, Any]: return eval_data

    # ------------------- Private Helper for HTML (Upgraded) ------------------- #
    
    def _create_html_artifact(self, data_to_print: Any, trace_config: Dict[str, Any], method_name: str) -> str:
        # ... (file naming logic is unchanged) ...
        addon_name = self.__class__.__name__
        grand_func_name = trace_config.get('grand_function_name', 'pipeline')
        log_dir = trace_config['log_dir']
        ts = datetime.datetime.now().strftime("%H%M%S")
        file_name = f"{grand_func_name}_{addon_name}_{method_name}_{ts}.html"
        file_path = os.path.join(log_dir, file_name)

        title = "Artifact"
        html_content = ""

        # --- ✨ CHANGE 1: Handle raw HTML strings ---
        # If the string looks like HTML, use it directly.
        if isinstance(data_to_print, str) and data_to_print.strip().startswith("<"):
            title = "HTML Artifact"
            html_content = data_to_print
        # --- End Change ---
        
        elif HAS_MATPLOTLIB and isinstance(data_to_print, Figure):
            # ... (matplotlib logic) ...
            title = "Figure Artifact"
            try:
                buf = io.BytesIO()
                data_to_print.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                img_data = base64.b64encode(buf.read()).decode('utf-8')
                html_content = f'<img src="data:image/png;base64,{img_data}" alt="matplotlib figure"/>'
            except Exception as e:
                html_content = f"<pre>Error rendering matplotlib figure: {e}</pre>"

        elif HAS_PLOTLY and isinstance(data_to_print, PlotlyFigure):
            # ... (plotly logic) ...
            title = "Plotly Figure Artifact"
            try:
                html_content = data_to_print.to_html(full_html=False, include_plotlyjs='cdn')
            except Exception as e:
                html_content = f"<pre>Error rendering Plotly figure: {e}</pre>"
        
        elif isinstance(data_to_print, (pd.DataFrame, pd.Series)):
            title = "DataFrame Artifact"
            html_content = data_to_print.to_html(classes='table', border=0, justify='left', index=True)
            
        elif isinstance(data_to_print, str):
            title = "String Artifact"
            html_content = f"<pre>{data_to_print}</pre>"

        elif isinstance(data_to_print, (dict, list)):
            title = "JSON/Dict/List Artifact"
            html_content = f"<pre>{json.dumps(data_to_print, indent=2, default=str)}</pre>"
            
        else:
            title = "Object Artifact"
            html_content = f"<pre>{str(data_to_print)}</pre>"
            
        # ... (full_html template is unchanged) ...
        full_html = f"""
        <html><head><title>{title}</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; }} table {{ border-collapse: collapse; border: 1px solid #ccc; }}
            th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }} th {{ background-color: #f4f4f4; }}
            pre {{ background-color: #eee; padding: 10px; border-radius: 5px; border: 1px solid #ddd; white-space: pre-wrap; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
        </style></head>
        <body><h2>{title}</h2>
        <p><strong>Add-On:</strong> {addon_name}<br><strong>Method:</strong> {method_name}</p>
        <hr>
        {html_content}
        </body></html>
        """
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(full_html)
            
        return pathlib.Path(file_path).as_uri() # Returns the raw file:///... link

    def _set_trace_message(self, pipeline_extra_info: Dict[str, Any], message: str) -> None:
        if pipeline_extra_info is not None:
            pipeline_extra_info['current_trace_message'] = message
        else:
            print("Warning: _set_trace_message called but pipeline_extra_info is missing.")

    def set_attachment(self, data: Any):
        self._current_attachment = data

    def add_trace_print(self, pipeline_extra_info: Dict[str, Any], message: str) -> None:
        if ATTACHMENT_PLACEHOLDER not in message:
            self._set_trace_message(pipeline_extra_info, message)
            return

        if self._current_attachment is None:
            final_message = message.replace(ATTACHMENT_PLACEHOLDER, "[No Attachment Set]")
            self._set_trace_message(pipeline_extra_info, final_message)
            return

        try:
            trace_config = pipeline_extra_info.get('_trace_config')
            if not trace_config or not trace_config.get('log_dir'):
                 raise ValueError("No log_dir found in _trace_config")

            method_name = inspect.stack()[1].function
            
            file_url = self._create_html_artifact(
                self._current_attachment, 
                trace_config, 
                method_name
            )
            
            # --- ✨ CHANGE 2: Replace placeholder with the raw file URL ---
            final_message = message.replace(ATTACHMENT_PLACEHOLDER, file_url)
            # --- End Change ---
            
            self._set_trace_message(pipeline_extra_info, final_message)

        except Exception as e:
            final_message = message.replace(ATTACHMENT_PLACEHOLDER, f"[Artifact Error: {e}]")
            self._set_trace_message(pipeline_extra_info, final_message)
        
        finally:
            self._current_attachment = None