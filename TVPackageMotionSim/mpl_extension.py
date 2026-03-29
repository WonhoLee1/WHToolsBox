# -*- coding: utf-8 -*-
"""
[WHTOOLS] Matplotlib Advanced Extension Module (v1.0)
데이터 분석, 인터랙티브 어노테이션 및 다중 포맷 익스포트를 지원하는 Matplotlib 확장 라이브러리입니다.
재사용성을 고려하여 독립된 모듈로 설계되었습니다.
"""

import os
import csv
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt

class WHToolsPlotManager:
    """
    Matplotlib Figure에 인터랙티브 분석 및 데이터 추출 기능을 부여하는 매니저 클래스.
    """
    def __init__(self, parent_tk=None):
        self.parent_tk = parent_tk
        self.active_figs = []
        self._annotations = {} # fig별 고정 어노테이션 저장 리스트
        self._hover_labels = {} # fig별 현재 호버링 가이드 레이블

    def attach_interactivity(self, fig):
        """
        주어진 Figure에 호버 안내 및 클릭 마킹 기능을 연결합니다.
        """
        if fig in self.active_figs: return
        self.active_figs.append(fig)
        self._annotations[fig] = []
        
        # 이벤트 연결
        # 이벤트 연결
        fig.canvas.mpl_connect('motion_notify_event', lambda e: self._on_hover(e, fig))
        fig.canvas.mpl_connect('button_press_event', lambda e: self._on_click(e, fig))
        
        # [NEW] 개별 창 메뉴바 추가 (Export 및 기타 도구)
        self._add_custom_menu(fig)

    def _add_custom_menu(self, fig):
        """Matplotlib 백엔드에 따라 윈도우 메뉴바에 WHTOOLS 기능을 추가합니다."""
        backend = plt.get_backend().lower()
        
        if 'tkagg' in backend:
            self._setup_tk_menu(fig)
        elif 'qtagg' in backend or 'qt5agg' in backend:
            self._setup_qt_menu(fig)

    def _setup_tk_menu(self, fig):
        """Tkinter 기반 메뉴 구성"""
        try:
            import tkinter as tk
            window = fig.canvas.manager.window
            menubar = tk.Menu(window)
            
            # Export Menu
            export_menu = tk.Menu(menubar, tearoff=0)
            export_menu.add_command(label="Save as CSV (.csv)", command=lambda: self._export_action(fig, "csv"))
            export_menu.add_command(label="Save as TXT (.txt)", command=lambda: self._export_action(fig, "txt"))
            export_menu.add_separator()
            export_menu.add_command(label="Copy to Clipboard", command=lambda: self._export_action(fig, "clipboard"))
            
            # Tools Menu
            tools_menu = tk.Menu(menubar, tearoff=0)
            tools_menu.add_command(label="Clear All Annotations", command=lambda: self.clear_annotations(fig))
            tools_menu.add_command(label="Tight Layout", command=lambda: (fig.tight_layout(), fig.canvas.draw()))

            menubar.add_cascade(label="WHTOOLS: Export", menu=export_menu)
            menubar.add_cascade(label="WHTOOLS: Tools", menu=tools_menu)
            
            window.config(menu=menubar)
        except Exception as e:
            print(f"[WHTOOLS] Failed to add Tk menu: {e}")

    def _setup_qt_menu(self, fig):
        """Qt 기반 메뉴 구성 (PySide6/PyQt6)"""
        try:
            window = fig.canvas.manager.window
            menubar = window.menuBar()
            
            # Export Menu
            export_menu = menubar.addMenu("WHTOOLS: Export")
            export_menu.addAction("Save as CSV (.csv)", lambda: self._export_action(fig, "csv"))
            export_menu.addAction("Save as TXT (.txt)", lambda: self._export_action(fig, "txt"))
            export_menu.addSeparator()
            export_menu.addAction("Copy to Clipboard", lambda: self._export_action(fig, "clipboard"))
            
            # Tools Menu
            tools_menu = menubar.addMenu("WHTOOLS: Tools")
            tools_menu.addAction("Clear All Annotations", lambda: self.clear_annotations(fig))
            tools_menu.addAction("Tight Layout", lambda: (fig.tight_layout(), fig.canvas.draw()))
        except Exception as e:
            print(f"[WHTOOLS] Failed to add Qt menu: {e}")

    def _export_action(self, fig, fmt):
        """메뉴 항목 클릭 시 호출되는 익스포트 핸들러"""
        if fmt == "csv":
            path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="export_data.csv", title="Save as CSV")
            if path: PlotExporter.to_csv(fig, path)
        elif fmt == "txt":
            path = filedialog.asksaveasfilename(defaultextension=".txt", initialfile="export_data.txt", title="Save as TXT")
            if path: PlotExporter.to_txt(fig, path)
        elif fmt == "clipboard":
            # PlotExporter.to_clipboard requires parent_tk for clipboard access if not in a dialog
            # If parent_tk is None, we try to use fig's window if it's Tk
            p_tk = self.parent_tk
            if p_tk is None and hasattr(fig.canvas.manager, 'window'):
                p_tk = fig.canvas.manager.window
            
            if p_tk:
                PlotExporter.to_clipboard(fig, p_tk)
            else:
                # Fallback content return or generic message
                messagebox.showwarning("Export", "클립보드에 접근할 수 있는 부모 윈도우가 없습니다.")

    def clear_annotations(self, fig):
        """해당 Figure의 모든 고정 마킹을 제거합니다."""
        if fig in self._annotations:
            for ann in self._annotations[fig]:
                ann.remove()
            self._annotations[fig] = []
            fig.canvas.draw_idle()

    def _on_hover(self, event, fig):
        """마우스 호버 시 가장 가까운 점에 대한 안내 레이블 표시"""
        if not event.inaxes:
            if fig in self._hover_labels and self._hover_labels[fig]:
                self._hover_labels[fig].set_visible(False)
                fig.canvas.draw_idle()
            return

        ax = event.inaxes
        closest_line = None
        min_dist = float('inf')
        closest_point = None

        for line in ax.get_lines():
            # [Shapely 없이 픽셀 거리 기반 최단점 탐색]
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            if len(xdata) == 0: continue
            
            # 데이터 좌표를 픽셀(Display) 좌표로 변환
            points = ax.transData.transform(np.column_stack([xdata, ydata]))
            dists = np.sqrt((points[:, 0] - event.x)**2 + (points[:, 1] - event.y)**2)
            
            idx = np.argmin(dists)
            if dists[idx] < min_dist:
                min_dist = dists[idx]
                closest_line = line
                closest_point = (xdata[idx], ydata[idx])

        # 20픽셀 이내일 때만 안내 표시
        if min_dist < 20 and closest_point:
            if fig not in self._hover_labels:
                self._hover_labels[fig] = ax.annotate(
                    "", xy=(0,0), xytext=(20,20), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w", alpha=0.8),
                    arrowprops=dict(arrowstyle="->"))
            
            ann = self._hover_labels[fig]
            ann.set_visible(True)
            ann.xy = closest_point
            ann.set_text(f"x: {closest_point[0]:.4f}\ny: {closest_point[1]:.4f}")
            fig.canvas.draw_idle()
        else:
            if fig in self._hover_labels:
                self._hover_labels[fig].set_visible(False)
                fig.canvas.draw_idle()

    def _on_click(self, event, fig):
        """클릭 시 현재 호버링 중인 지점을 고정 마킹 (더블클릭 등 활용 가능)"""
        if event.button == 1 and fig in self._hover_labels:
            ann_guide = self._hover_labels[fig]
            if ann_guide.get_visible():
                # 새로운 고정 어노테이션 생성
                new_ann = event.inaxes.annotate(
                    ann_guide.get_text(), xy=ann_guide.xy, xytext=(20, 20),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="#ffeaa7", alpha=0.9),
                    arrowprops=dict(arrowstyle="->", color="red"))
                self._annotations[fig].append(new_ann)
                fig.canvas.draw_idle()

class PlotExporter:
    """
    Matplotlib Figure의 커브 데이터를 다양한 형식으로 내보내는 유틸리티 클래스.
    """
    @staticmethod
    def get_curve_data(fig):
        """Figure 내의 모든 Axes와 Line 데이터를 딕셔너리 형태로 추출합니다."""
        data_packet = []
        for i, ax in enumerate(fig.axes):
            ax_name = ax.get_title() or f"Axes_{i}"
            for line in ax.get_lines():
                label = line.get_label() or "unlabeled"
                x = line.get_xdata()
                y = line.get_ydata()
                data_packet.append({
                    "axes": ax_name,
                    "label": label,
                    "x": x,
                    "y": y
                })
        return data_packet

    @staticmethod
    def to_csv(fig, filepath):
        """데이터를 CSV 파일로 저장합니다."""
        data_list = PlotExporter.get_curve_data(fig)
        with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            for entry in data_list:
                writer.writerow([f"## Axes: {entry['axes']} | Label: {entry['label']}"])
                writer.writerow(["X_Value", "Y_Value"])
                for vx, vy in zip(entry['x'], entry['y']):
                    writer.writerow([vx, vy])
                writer.writerow([]) # 공백 라인

    @staticmethod
    def to_txt(fig, filepath):
        """데이터를 텍스트 파일로 저장합니다."""
        content = PlotExporter.to_formatted_string(fig)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

    @staticmethod
    def to_clipboard(fig, parent_tk):
        """데이터를 클립보드에 복사합니다."""
        content = PlotExporter.to_formatted_string(fig)
        parent_tk.clipboard_clear()
        parent_tk.clipboard_append(content)
        messagebox.showinfo("Export", "데이터가 클립보드에 복사되었습니다.")

    @staticmethod
    def to_formatted_string(fig):
        """데이터를 읽기 쉬운 텍스트 형식의 문자열로 변환합니다."""
        lines = [f"=== [WHTOOLS] Data Export: {fig.canvas.manager.get_window_title()} ==="]
        data_list = PlotExporter.get_curve_data(fig)
        for entry in data_list:
            lines.append(f"\n[Axes] {entry['axes']} | [Curve] {entry['label']}")
            lines.append("-" * 40)
            lines.append(f"{'X_Value':>15} | {'Y_Value':>15}")
            # 너무 많은 데이터는 요약 (상위 20개, 하위 20개 등) - 여기서는 전체 출력
            for vx, vy in zip(entry['x'], entry['y']):
                lines.append(f"{vx:15.6f} | {vy:15.6f}")
        return "\n".join(lines)

class ExportDialog(tk.Toplevel):
    """
    열려 있는 여러 Figure 중 익스포트 대상을 선택하는 팝업 대화상자.
    """
    def __init__(self, parent, plot_manager):
        super().__init__(parent)
        self.parent = parent
        self.pm = plot_manager
        self.title("WHTOOLS: Data Export Manager")
        self.geometry("450x400")
        self.selected_indices = []
        
        # Figure 리스트 확보
        self.figs = [plt.figure(i) for i in plt.get_fignums()]
        
        self._build_ui()
        self.grab_set()

    def _build_ui(self):
        main_f = ttk.Frame(self, padding=10)
        main_f.pack(fill="both", expand=True)

        ttk.Label(main_f, text="내보낼 그래프 창을 선택하세요:", font=("Arial", 10, "bold")).pack(anchor="w")
        
        # 목록 상자
        list_f = ttk.Frame(main_f)
        list_f.pack(fill="both", expand=True, pady=5)
        
        self.lb = tk.Listbox(list_f, selectmode="multiple", font=("Consolas", 9))
        self.lb.pack(side="left", fill="both", expand=True)
        
        sb = ttk.Scrollbar(list_f, orient="vertical", command=self.lb.yview)
        sb.pack(side="right", fill="y")
        self.lb.config(yscrollcommand=sb.set)

        for fig in self.figs:
            title = fig.canvas.manager.get_window_title() or f"Figure {fig.number}"
            self.lb.insert(tk.END, title)

        # 버튼 영역
        btn_f = ttk.Frame(main_f)
        btn_f.pack(fill="x", pady=10)
        
        ttk.Button(btn_f, text="CSV로 저장", command=lambda: self._exec_export("csv")).pack(side="left", padx=2)
        ttk.Button(btn_f, text="TXT로 저장", command=lambda: self._exec_export("txt")).pack(side="left", padx=2)
        ttk.Button(btn_f, text="클립보드로 복사", command=lambda: self._exec_export("clipboard")).pack(side="left", padx=2)
        ttk.Button(btn_f, text="닫기", command=self.destroy).pack(side="right", padx=2)

    def _exec_export(self, fmt):
        indices = self.lb.curselection()
        if not indices:
            messagebox.showwarning("Warning", "내보낼 항목을 선택하세요.")
            return

        for idx in indices:
            fig = self.figs[idx]
            if fmt == "csv":
                path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=f"export_data_{idx}.csv")
                if path: PlotExporter.to_csv(fig, path)
            elif fmt == "txt":
                path = filedialog.asksaveasfilename(defaultextension=".txt", initialfile=f"export_data_{idx}.txt")
                if path: PlotExporter.to_txt(fig, path)
            elif fmt == "clipboard":
                PlotExporter.to_clipboard(fig, self.parent)
        
        if fmt != "clipboard":
            messagebox.showinfo("Success", "데이터 내보내기가 완료되었습니다.")
        self.destroy()

def apply_whtools_extension(fig, parent_tk=None):
    """
    [Shorthand] 한 줄의 코드로 모든 확장 기능을 Figure에 적용합니다.
    """
    manager = WHToolsPlotManager(parent_tk)
    manager.attach_interactivity(fig)
    return manager
