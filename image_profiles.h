#pragma once

#include <DGtal/helpers/StdDefs.h>
#include <DGtal/images/ImageContainerBySTLVector.h>
#include <DGtal/images/ConstImageAdapter.h>
#include <DGtal/kernel/BasicPointFunctors.h>
#include <DGtal/io/writers/PGMWriter.h>

using namespace DGtal;
using namespace Z3i;

using Image3D = ImageContainerBySTLVector<Z3i::Domain, unsigned int>;
using Image2D = ImageContainerBySTLVector<Z2i::Domain, unsigned char>;
using ImageAdapterExtractor = DGtal::ConstImageAdapter<
    Image3D, Z2i::Domain,
    DGtal::functors::Point2DEmbedderIn3D<Z3i::Domain>,
    Image3D::Value,
    DGtal::functors::Identity>;

Image2D 
compute2D_profile(const RealPoint &normalDir, double widthImageScan,
                                  const Image3D &meshVolImage,
                                  const Image2D::Domain &aDomain2D, int maxScan) {
    Point ptC = (meshVolImage.domain().lowerBound() + meshVolImage.domain().upperBound()) / 2;
    Point center = moveCenterAlongDirection(ptC, -normalDir, maxScan);

    Image2D result(aDomain2D);
    for (auto it = result.domain().begin(); it != result.domain().end(); ++it) result.setValue(*it, 0);

    DGtal::functors::Point2DEmbedderIn3D<Z3i::Domain> embedder(
        meshVolImage.domain(), center, normalDir, widthImageScan);
        
    DGtal::functors::Identity idV;
    ImageAdapterExtractor extractedImage(meshVolImage, aDomain2D, embedder, idV);

    int k = 0;
    bool firstFound = false;

    while (k < maxScan || !firstFound) {
        embedder.shiftOriginPoint(normalDir);
        for (auto it = extractedImage.domain().begin(); it != extractedImage.domain().end(); ++it) {
            if (result(*it) == 0 && extractedImage(*it) != 0) {
                result.setValue(*it, 255);
                if (!firstFound) firstFound = true;
            }
        }
        if (firstFound) ++k;
    }

    return result;
}

Image2D 
compute2D_profile(const RealPoint &normalDir, Z3i::RealPoint secDir, double widthImageScan,
                                  const Image3D &meshVolImage,
                                  const Image2D::Domain &aDomain2D, int maxScan){

    Point ptC = (meshVolImage.domain().lowerBound() + meshVolImage.domain().upperBound()) / 2;
    Point center = moveCenterAlongDirection(ptC, -normalDir, maxScan);
   
    //domaine de l'image de profile
    Image2D result(aDomain2D);
    //init all value to 0
    for (auto it = result.domain().begin(); it != result.domain().end(); ++it) result.setValue(*it, 0);

    //embedder 3D to 2D (loop over 3d volume by image loop)
    DGtal::functors::Point2DEmbedderIn3D<DGtal::Z3i::Domain >  embedder(
        meshVolImage.domain(), center, normalDir, secDir, widthImageScan);
    //identity (but we can make rotate or translate here)
    DGtal::functors::Identity idV;
    //extractor 3d to 2D
    ImageAdapterExtractor extractedImage(meshVolImage, aDomain2D, embedder, idV);

    int k = 0;
    bool firstFound = false;

    while (k < maxScan || !firstFound) {
        embedder.shiftOriginPoint(normalDir);
        for (auto it = extractedImage.domain().begin(); it != extractedImage.domain().end(); ++it) {
            if (result(*it) == 0 && extractedImage(*it) != 0) {
                result.setValue(*it, 255);
                if (!firstFound) firstFound = true;
            }
        }
        if (firstFound) ++k;
    }
 

    return result;
}