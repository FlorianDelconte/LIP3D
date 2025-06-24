#ifndef CUSTOM_VIEWER_3D_H
#define CUSTOM_VIEWER_3D_H

#include <DGtal/io/viewers/Viewer3D.h>
#include <DGtal/io/Color.h>
#include <DGtal/helpers/StdDefs.h>
#include <QKeyEvent>
#include <QString>
#include <fstream>
#include <string>

class CustomViewer3D : public DGtal::Viewer3D<>
{
public:
    CustomViewer3D();

    void changeDefaultBGColor(const DGtal::Color &col);
    void saveCameraPosition(const std::string &filename);
    void loadCameraPosition(const std::string &filename);



    std::string myInfoDisplay = "No information loaded...";
    bool myIsDisplayingInfoMode = false;
    DGtal::Z3i::RealPoint centerMesh;

protected:
    virtual void init() override;
    virtual void keyPressEvent(QKeyEvent *e) override;
};

#endif // CUSTOM_VIEWER_3D_H
