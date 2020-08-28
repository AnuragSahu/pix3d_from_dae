import bpy

#location_file = open("../cam_locations/box_locations.txt","w+")
location_file = open("../cam_locations/rack_locations.txt","w+")

object = bpy.data.objects["BoxA"]#bpy.data.objects['Camera']
for f in range(bpy.context.scene.frame_start,bpy.context.scene.frame_end):
    bpy.context.scene.frame_set(f)
    location = str(object.location.x) + str(" ")+ str(object.location.y)+ str(" ")+ str(object.location.z) +str(" ")
    angles = str(object.rotation_euler.x) + str(" ")+ str(object.rotation_euler.y)+ str(" ")+ str(object.rotation_euler.z) +str("\n")
    location_file.write(location)
    location_file.write(angles)
    
location_file.close()

