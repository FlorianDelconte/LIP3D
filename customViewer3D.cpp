#include "customViewer3D.h"
#include <sstream>

CustomViewer3D::CustomViewer3D() : Viewer3D<>() {}

void CustomViewer3D::init()
{
    Viewer3D<>::init();
    Viewer3D<>::setKeyDescription(Qt::Key_I, "Display mesh informations about #faces, #vertices");
    Viewer3D<>::setKeyDescription(Qt::CTRL + Qt::Key_J, "Save camera position");
    Viewer3D<>::setKeyDescription(Qt::CTRL + Qt::Key_K, "Load camera position");
    Viewer3D<>::setGLDoubleRenderingMode(false);
}

void CustomViewer3D::keyPressEvent(QKeyEvent *e)
{
    bool handled = false;

    if (e->key() == Qt::Key_I)
    {
        handled = true;
        myIsDisplayingInfoMode = !myIsDisplayingInfoMode;
        std::stringstream ss;
        qglviewer::Vec camPos = camera()->position();
        DGtal::Z3i::RealPoint c(camPos[0], camPos[1], camPos[2]);
        ss << myInfoDisplay << " distance to camera: " << (c - centerMesh).norm();
        Viewer3D<>::displayMessage(QString(myIsDisplayingInfoMode ? ss.str().c_str() : " "), 1000000);
        Viewer3D<>::update();
    }
    else if (e->key() == Qt::Key_J && (e->modifiers() & Qt::ControlModifier))
    {
        handled = true;
        saveCameraPosition("camera_pos.txt");
        Viewer3D<>::displayMessage("Camera position saved.", 3000);
    }
    else if (e->key() == Qt::Key_K && (e->modifiers() & Qt::ControlModifier))
    {
        handled = true;
        loadCameraPosition("camera_pos.txt");
        Viewer3D<>::displayMessage("Camera position loaded.", 3000);
        Viewer3D<>::update();
    }

    if (!handled)
        Viewer3D<>::keyPressEvent(e);
}

void CustomViewer3D::changeDefaultBGColor(const DGtal::Color &col)
{
    myDefaultBackgroundColor = col;
    Viewer3D<>::update();
    Viewer3D<>::draw();
}

void CustomViewer3D::saveCameraPosition(const std::string &filename)
{
    std::ofstream out(filename);
    if (!out) return;

    qglviewer::Vec pos = camera()->position();
    qglviewer::Vec dir = camera()->viewDirection();
    qglviewer::Vec up = camera()->upVector();
    out << pos.x << " " << pos.y << " " << pos.z << "\n";
    out << dir.x << " " << dir.y << " " << dir.z << "\n";
    out << up.x << " " << up.y << " " << up.z << "\n";
}

void CustomViewer3D::loadCameraPosition(const std::string &filename)
{
    std::ifstream in(filename);
    if (!in) return;

    qglviewer::Vec pos, dir, up;
    in >> pos.x >> pos.y >> pos.z;
    in >> dir.x >> dir.y >> dir.z;
    in >> up.x >> up.y >> up.z;

    camera()->setPosition(pos);
    camera()->setViewDirection(dir);
    camera()->setUpVector(up);
}

