from FileService import add_xmls_in_renate
from AnnotationService import create_and_add_annotations
from AuthService import get_authenticated_session


COLLECTION_ID = "477883a2-8ca5-462e-b496-8a18f4500958"


def main():
    session = get_authenticated_session()

    print("[INFO] Converting PDFs to JATS XML...")
    add_xmls_in_renate(session, COLLECTION_ID)

    print("[INFO] Generating annotations of physics articles using ML model...")
    create_and_add_annotations(session, COLLECTION_ID)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()