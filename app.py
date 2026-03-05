from FileService import add_xmls_in_renate
from AnnotationService import create_and_add_annotations
from AuthService import get_authenticated_session


s = get_authenticated_session()
add_xmls_in_renate(s, "477883a2-8ca5-462e-b496-8a18f4500958")


# considering only physics collection
print("Genetating annotations of physics articles using ML model")
create_and_add_annotations(s, "477883a2-8ca5-462e-b496-8a18f4500958")
