"""
This module includes all the plotting functions, including 2D, 3D, and dot plots.

The module provides various functions for plotting and visualizing data in 2D and 3D,
as well as functions for creating interactive plots using Plotly.

Functions:
    - plot_2D: Plots a 2D field using Matplotlib.
    - plot_scatter_2D: Creates a 2D scatter plot using Matplotlib.
    - plot_plane_go: Plots a cross-section XY plane in 3D using Plotly.
    - plot_3D_dots_go: Plots 3D dots interactively in the browser using Plotly.
    - plot_3D_density: Plots 3D density in the browser using Plotly.
    - plot_scatter_3D: Plots 3D scatter points using Matplotlib.
    - box_set_go: Sets up a 3D plot with box boundaries using Plotly.
    - plotDots: Plots an array of dots interactively in the browser using Plotly.
"""
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import extra_functions_package.functions_general as fg

# standard values for fonts
ticksFontSize = 18
xyLabelFontSize = 20
legendFontSize = 20


def plot_2D(field, x=None, y=None, xname='', yname='', map='jet', vmin=None, vmax=None, title='',
            ticksFontSize=ticksFontSize, xyLabelFontSize=xyLabelFontSize, grid=False,
            axis_equal=False, xlim=None, ylim=None, ax=None, show=True, ijToXY=True, origin='lower',
            interpolation='bilinear',
            **kwargs) -> object:
    """
        Plots a 2D field using Matplotlib.

        Parameters:
            field: 2D array to plot.
            x, y: Coordinates for the axes.
            xname, yname: Labels for the x and y axes.
            map: Colormap to use.
            vmin, vmax: Minimum and maximum values for color scaling.
            title: Title of the plot.
            ticksFontSize: Font size for ticks.
            xyLabelFontSize: Font size for axis labels.
            grid: Boolean to show grid.
            axis_equal: Boolean to set equal scaling for the axes.
            xlim, ylim: Limits for the x and y axes.
            ax: Matplotlib axis object.
            show: Boolean to show the plot.
            ijToXY: Boolean to transpose the field.
            origin: Origin of the plot.
            interpolation: Interpolation method for imshow.
            **kwargs: Additional keyword arguments for imshow.

        Returns:
            Matplotlib axis object.
        """
    fieldToPlot = field
    if ijToXY:
        origin = 'lower'
        fieldToPlot = np.copy(field).transpose()
    if x is None:
        x = range(np.shape(fieldToPlot)[0])
    if y is None:
        y = range(np.shape(fieldToPlot)[1])
    if ax is None:
        if axis_equal:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(fieldToPlot,
                       interpolation=interpolation, cmap=map,
                       origin=origin, aspect='auto',  # aspect ration of the axes
                       extent=[x[0], x[-1], y[0], y[-1]],
                       vmin=vmin, vmax=vmax, label='sdfsd', **kwargs)
    cbr = plt.colorbar(image, ax=ax, shrink=0.8, pad=0.02, fraction=0.1)
    cbr.ax.tick_params(labelsize=ticksFontSize)
    plt.xticks(fontsize=ticksFontSize)
    plt.yticks(fontsize=ticksFontSize)
    ax.set_xlabel(xname, fontsize=xyLabelFontSize)
    ax.set_ylabel(yname, fontsize=xyLabelFontSize)
    plt.title(title, fontweight="bold", fontsize=26)
    if axis_equal:
        ax.set_aspect('equal', adjustable='box')
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.tight_layout()
    plt.grid(grid)
    if show:
        plt.show()
    return ax


def plot_scatter_2D(x, y, xname='', yname='', title='',
                    ticksFontSize=ticksFontSize, xyLabelFontSize=xyLabelFontSize,
                    axis_equal=False, xlim=None, ylim=None, ax=None, show=True,
                    size=plt.rcParams['lines.markersize'] ** 2, color=None,
                    **kwargs):
    """
    Creates a 2D scatter plot using Matplotlib.

    Parameters:
        x, y: Data for the x and y axes.
        xname, yname: Labels for the x and y axes.
        title: Title of the plot.
        ticksFontSize: Font size for ticks.
        xyLabelFontSize: Font size for axis labels.
        axis_equal: Boolean to set equal scaling for the axes.
        xlim, ylim: Limits for the x and y axes.
        ax: Matplotlib axis object.
        show: Boolean to show the plot.
        size: Size of the scatter points.
        color: Color of the scatter points.
        **kwargs: Additional keyword arguments for scatter.

    Returns:
        Matplotlib axis object.
    """
    if ax is None:
        if axis_equal:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
    plt.scatter(x, y, s=size, color=color, **kwargs)
    plt.xticks(fontsize=ticksFontSize)
    plt.yticks(fontsize=ticksFontSize)
    ax.set_xlabel(xname, fontsize=xyLabelFontSize)
    ax.set_ylabel(yname, fontsize=xyLabelFontSize)
    plt.title(title, fontweight="bold", fontsize=26)
    if axis_equal:
        ax.set_aspect('equal', adjustable='box')
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.tight_layout()
    if show:
        plt.show()
    return ax


def plot_plane_go(z, mesh, fig=None, opacity=0.6, show=False,
                  colorscale=([0, '#aa9ce2'], [1, '#aa9ce2']), **kwargs):
    """
    Plots a cross-section XY plane in 3D using Plotly.

    Parameters:
        z: Z coordinate of the plane.
        mesh: Meshgrid for the coordinates.
        fig: Plotly figure object.
        opacity: Opacity of the plane.
        show: Boolean to show the plot.
        colorscale: Colorscale for the plane.
        **kwargs: Additional keyword arguments for go.Surface.

    Returns:
        Plotly figure object.
    """
    xyz = fg.arrays_from_mesh(mesh)

    if fig is None:
        fig = go.Figure()
    fig.add_trace(go.Surface(x=xyz[0], y=xyz[1], z=z,
                             opacity=opacity, colorscale=colorscale, showscale=False, **kwargs))
    if show:
        fig.show()
    return fig


def plot_3D_dots_go(dots, mode='markers', marker=None, fig=None, show=False, **kwargs):
    """
    Plots 3D dots interactively in the browser using Plotly.

    Parameters:
        dots: Array of dot coordinates.
        mode: Plotting mode (e.g., 'markers').
        marker: Marker style.
        fig: Plotly figure object.
        show: Boolean to show the plot.
        **kwargs: Additional keyword arguments for go.Scatter3d.

    Returns:
        Plotly figure object.
    """
    if marker is None:
        marker = {'size': 8, 'color': 'black'}
    if fig is None:
        fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=dots[:, 0], y=dots[:, 1], z=dots[:, 2],
                               mode=mode, marker=marker, **kwargs))
    if show:
        fig.show()
    return fig


def plot_3D_density(E, resDecrease=(1, 1, 1), mesh=None,
                    xMinMax=None, yMinMax=None, zMinMax=None,
                    surface_count=20, show=True,
                    opacity=0.5, colorscale='RdBu',
                    opacityscale=None, fig=None,  scaling=None, **kwargs):
    """
    Plots 3D density in the browser using Plotly.

    Parameters:
        E: 3D array to plot.
        resDecrease: Resolution decrease factors.
        mesh: Meshgrid for the coordinates.
        xMinMax, yMinMax, zMinMax: Boundaries for the x, y, and z axes.
        surface_count: Number of layers to show.
        show: Boolean to show the plot.
        opacity: Opacity of the surfaces.
        colorscale: Colorscale for the plot.
        opacityscale: Custom opacity scale.
        fig: Plotly figure object.
        scaling: Scaling factors for the coordinates.
        **kwargs: Additional keyword arguments for go.Volume.

    Returns:
        Plotly figure object.
    """
    if mesh is None:
        shape = np.array(np.shape(E))
        if resDecrease is not None:
            shape = (shape // resDecrease)
        if zMinMax is None:
            zMinMax = [0, shape[0]]
        if yMinMax is None:
            yMinMax = [0, shape[1]]
        if xMinMax is None:
            xMinMax = [0, shape[2]]

        X, Y, Z = np.mgrid[
                  xMinMax[0]:xMinMax[1]:shape[0] * 1j,
                  yMinMax[0]:yMinMax[1]:shape[1] * 1j,
                  zMinMax[0]:zMinMax[1]:shape[2] * 1j
                  ]
    else:
        X, Y, Z = mesh
    if scaling is not None:
        X *= scaling[0]
        Y *= scaling[1]
        Z *= scaling[2]
    values = E[::resDecrease[0], ::resDecrease[1], ::resDecrease[2]]
    if fig is None:
        fig = go.Figure()
    fig.add_trace(go.Volume(
        x=X.flatten(),  # collapsed into 1 dimension
        y=Y.flatten(),
        z=Z.flatten(),
        value=values.flatten(),
        isomin=values.min(),
        isomax=values.max(),
        opacity=opacity,  # needs to be small to see through all surfaces
        opacityscale=opacityscale,
        surface_count=surface_count,  # needs to be a large number for good volume rendering
        colorscale=colorscale,
        **kwargs
    ))
    if show:
        fig.show()
    return fig


def plot_scatter_3D(X, Y, Z, ax=None, size=plt.rcParams['lines.markersize'] ** 2, color=None,
                    viewAngles=(70, 0), show=True, **kwargs):
    """
    Plots 3D scatter points using Matplotlib.

    Parameters:
        X, Y, Z: Coordinates of the scatter points.
        ax: Matplotlib axis object.
        size: Size of the scatter points.
        color: Color of the scatter points.
        viewAngles: Viewing angles for the plot.
        show: Boolean to show the plot.
        **kwargs: Additional keyword arguments for scatter.

    Returns:
        Matplotlib axis object.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, s=size, color=color, **kwargs)  # plot the point (2,3,4) on the figure
    ax.view_init(*viewAngles)
    if show:
        plt.show()
    return ax


def box_set_go(fig, xyzMinMax=(-1, 1, -1, 1, -1, 1), width=3, perBox=0, mesh=None, autoDots=None,
               return_boundaries=False, aspects=(2, 2, 2), lines=True):
    """
    Sets up a 3D plot with box boundaries using Plotly.

    Parameters:
        fig: Plotly figure object.
        xyzMinMax: Boundaries for the x, y, and z axes.
        width: Width of the box lines.
        perBox: Percentage to make the box bigger.
        mesh: Meshgrid for the coordinates.
        autoDots: Dots to use for automatic boundary detection.
        return_boundaries: Boolean to return the boundaries.
        aspects: Aspect ratios for the axes.
        lines: Boolean to draw lines.

    Returns:
        Plotly figure object, and boundaries if return_boundaries is True.
    """
    if autoDots is not None:
        dots = autoDots
        xMin, xMax = 1e10, 0
        yMin, yMax = 1e10, 0
        zMin, zMax = 1e10, 0
        for dot in dots:
            if dot[0] < xMin:
                xMin = dot[0]
            if dot[0] > xMax:
                xMax = dot[0]
            if dot[1] < yMin:
                yMin = dot[1]
            if dot[1] > yMax:
                yMax = dot[1]
            if dot[2] < zMin:
                zMin = dot[2]
            if dot[2] > zMax:
                zMax = dot[2]
    elif mesh is not None:
        xyz = fg.arrays_from_mesh(mesh)
        xMin, xMax = xyz[0][0], xyz[0][-1]
        yMin, yMax = xyz[1][0], xyz[1][-1]
        zMin, zMax = xyz[2][0], xyz[2][-1]
    else:
        xMin, xMax = xyzMinMax[0], xyzMinMax[1]
        yMin, yMax = xyzMinMax[2], xyzMinMax[3]
        zMin, zMax = xyzMinMax[4], xyzMinMax[5]
    if perBox != 0:
        xMin, xMax = xMin - (xMax - xMin) * perBox, xMax + (xMax - xMin) * perBox
        yMin, yMax = yMin - (yMax - yMin) * perBox, yMax + (yMax - yMin) * perBox
        zMin, zMax = zMin - (zMax - zMin) * perBox, zMax + (zMax - zMin) * perBox
    if lines:
        lineX = np.array([[xMin, yMin, zMin], [xMax, yMin, zMin], [xMax, yMax, zMin],
                          [xMin, yMax, zMin], [xMin, yMin, zMin]])
        plot_3D_dots_go(lineX, fig=fig, mode='lines', line={'width': width, 'color': 'black'})
        lineX = np.array([[xMin, yMin, zMax], [xMax, yMin, zMax], [xMax, yMax, zMax],
                          [xMin, yMax, zMax], [xMin, yMin, zMax]])
        plot_3D_dots_go(lineX, fig=fig, mode='lines', line={'width': width, 'color': 'black'})
        lineX = np.array([[xMin, yMin, zMin], [xMin, yMin, zMax]])
        plot_3D_dots_go(lineX, fig=fig, mode='lines', line={'width': width, 'color': 'black'})
        lineX = np.array([[xMax, yMin, zMin], [xMax, yMin, zMax]])
        plot_3D_dots_go(lineX, fig=fig, mode='lines', line={'width': width, 'color': 'black'})
        lineX = np.array([[xMax, yMax, zMin], [xMax, yMax, zMax]])
        plot_3D_dots_go(lineX, fig=fig, mode='lines', line={'width': width, 'color': 'black'})
        lineX = np.array([[xMin, yMax, zMin], [xMin, yMax, zMax]])
        plot_3D_dots_go(lineX, fig=fig, mode='lines', line={'width': width, 'color': 'black'})
    per = 0.01
    boundaries = [xMin - (xMax - xMin) * per, xMax + (xMax - xMin) * per,
                  yMin - (yMax - yMin) * per, yMax + (yMax - yMin) * per,
                  zMin - (zMax - zMin) * per, zMax + (zMax - zMin) * per]
    fig.update_layout(font_size=24, font_family="Times New Roman", font_color='black',
                      legend_font_size=20,
                      showlegend=False,

                      scene=dict(
                          # annotations=[dict(x=[ 2], y=[-2], z=[-.5], ax=-2, ay=-2), dict(align='left'),
                          #              dict(align='left')],
                          xaxis_title=dict(text='x', font=dict(size=45)),
                          yaxis_title=dict(text='y', font=dict(size=45)),
                          zaxis_title=dict(text='z', font=dict(size=45)),

                          aspectratio_x=aspects[0], aspectratio_y=aspects[1], aspectratio_z=aspects[2],

                          xaxis=dict(range=[xMin - (xMax - xMin) * per, xMax + (xMax - xMin) * per],
                                     showticklabels=False, zeroline=False,
                                     showgrid=False,  # gridcolor='white',
                                     showbackground=False  # backgroundcolor='white',
                                     ),
                          yaxis=dict(range=[yMin - (yMax - yMin) * per, yMax + (yMax - yMin) * per],
                                     showticklabels=False, zeroline=False,
                                     showgrid=False,  # gridcolor='white',
                                     showbackground=False
                                     ),
                          zaxis=dict(range=[zMin - (zMax - zMin) * per, zMax + (zMax - zMin) * per],
                                     showticklabels=False, zeroline=False,
                                     showgrid=False,  # gridcolor='white',
                                     showbackground=False
                                     ), ),  # range=[-0.5, 0.5],
                      )
    if return_boundaries:
        return fig, boundaries
    return fig


def plotDots(dots, dots_bound=None, show=True, color='black', size=15, width=185, fig=None,
             save=None):
    """
    Plots an array of dots interactively in the browser using Plotly.

    Parameters:
        dots: Array of dot coordinates.
        dots_bound: Dots to use for boundary detection.
        show: Boolean to show the plot.
        color: Color of the dots.
        size: Size of the dots.
        width: Width of the dot borders.
        fig: Plotly figure object.
        save: Path to save the plot.

    Returns:
        Plotly figure object.
    """
    colorLine = 'white'
    # colorLine = 'black'
    # print(dots)
    if len(dots) == 0 or len(dots) == 1:
        dots = np.array([[0, 0, 0]])
    if isinstance(dots, dict):
        dots = np.array([dot for dot in dots])
    if isinstance(dots_bound, dict):
        dots_bound = np.array([dot for dot in dots_bound])
    # print(dots)
    if dots_bound is None:
        dots_bound = dots
    if fig is None:
        fig = plot_3D_dots_go(dots, marker={'size': size, 'color': color,
                                            'line': dict(width=width, color=colorLine)})
    else:
        plot_3D_dots_go(dots, fig=fig, marker={'size': size, 'color': color,
                                               'line': dict(width=width, color=colorLine)})
    box_set_go(fig, mesh=None, autoDots=dots_bound, perBox=0.01)
    if save is not None:
        fig.write_html(save)
    if show:
        fig.show()
    return fig

