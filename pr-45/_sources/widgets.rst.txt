Topo & The Widgets!
=================

This document function to explain the widget modules in mom_bathy.

mom6_bathy comes with three UI modules/classes that wrap each of the main classes (vgrid, Grid, Topo) to help with the creation of vertical (VGridCreator) and horizontal (GridCreator) grids and the editing of Topography (TopoEditor).

The creators' function as a visual wrapper around the contructors of their respective classes with helpful sliders and visualizations. They create folders called "VgridLibrary" and "GridLibrary" to store the created grids and the current grid can be directly accessed as an object in each creator.


Topo & Topo Editor
------------------------
The Topo & TopoEditor is a bit more nuanced and will be explained further down here.

The grids are simple classes that are only touched once (creation time). Topo is built on top of the horizontal grid and is majorly edited during the development cycle (filling bays, deepening ridges, etc...).
As such, topo has a lot of edit functions. The TopoEditor is a visual tool to see the topo and wrap the edit functions in a "point-and-click" way. 

The editing functions are an ever expanding list, but as of writing, here is the list, with the ones in the Topo Editor starred:
1. * Edit depth at a specific point 
2. * Edit the minimum depth
3. * Erase a basin at a selected point
4. * Erase every basin but the basin with the selected point
5. Given a dataset containing land fraction, generate and apply ocean mask.
6. Apply a ridge to the bathymetry.

You can also just apply an initializer again, which is technically an edit (None of these are supported in the TopoEditor): 
1. Set flat bathy
2. Set spoon bathy
3. Set bowl bathy
4. Set from dataset 
5. Set from previous topo object

To support the iterative process, we provide undo & redo functionality (across sessions!) in the topography. To do this, there is a lot of overhead! Here's how it's setup.

We use a folder to support the topo object, and it contains all the history of edits you make with the topo object or the TopoEditor. It has four items:
1. The grid underlying the topo
2. The original topo (which is blank) 
3. A temporary command history for the *in session* changes
4. A permanent command history that applies across sessions. Temporary & Permanent only get synced when the topo is saved!

How it works!
1. You create your topo, which creates our folder of four files from above.
2. You make changes using the topo object or through the topo editor. 
3. The change gets added to the temporary command history, and whenever the object is saved, to the permanent command history.
4. The change then gets *commited*! That's right! 

Git and how to use it!
------------------------

We're using Git to implement undo/redo functionality in the topo object! You can see the log of changes by doing git log in your topo folder! 
The temporary command history is a json file corresponding to git commit sha's. So you can look up the details of the change from your git log in the command history file. 

We also implement a git create branch and checkout, which can be triggered with topo.tcm.create_branch("branchname") and topo.tcm.checkout("branchname").
You can also do a "tag" with topo.tcm.tag("tagname"), which will show up in your git history and save a topo file named like tagname_topog.nc. You cannot checkout the tag because that creates some undefined behavior we don't want!

To undo or redo, you can use topo.tcm.undo() or topo.tcm.redo(). It'll show up in your git log as REDO-sha or UNDO-sha of the sha that was undone or redone. Be careful not to undo your initial set function!

WARNING! Don't do git commands in your folder! Use the topo.tcm functions! Only use git log --oneline! We manage the topo independently (for now), and using something like a git checkout would break the functionality (probably).

Naunces (Initialization, naming, etc...)
--------------------------------------------

Folders are named after the hash of the grid tlon variable. This means that any initialization with the *same* grid will exist in the same folder. 

You can initilalize Topo in three ways:
1. In __init__, which is like Topo(). That will create an empty topo with whatever minimum depth you have (So if you have a preexiting topo with the same grid, it will add two edits, a minimum depth edit that you specify, and it will set your depth to NaN)
2. In class method from_topo