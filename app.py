from FileService import add_xmls_in_renate
from AnnotationService import create_and_add_annotations
from AuthService import get_authenticated_session


s = get_authenticated_session()
#add_xmls_in_renate(s, "10d7aac1-a01e-43f7-be42-8280f16160c5")

add_xmls_in_renate(s, "477883a2-8ca5-462e-b496-8a18f4500958") #production


#considering only physics collection
#create_and_add_annotations(s, "10d7aac1-a01e-43f7-be42-8280f16160c5")