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
                  const Image2D::Domain &aDomain2D)
{
    // Étape 1 : récupérer les bornes du domaine
    Z3i::Point ptMin = meshVolImage.domain().lowerBound();
    Z3i::Point ptMax = meshVolImage.domain().upperBound();

    // Étape 2 : centre géométrique du domaine (support de l'image)
    Z3i::Point ptC = (ptMin + ptMax) / 2;

    // Étape 3 : projection des coins pour calculer la profondeur nécessaire
    double projMin = ptMin[0]*normalDir[0] + ptMin[1]*normalDir[1] + ptMin[2]*normalDir[2];
    double projMax = ptMax[0]*normalDir[0] + ptMax[1]*normalDir[1] + ptMax[2]*normalDir[2];
    int maxScan = std::ceil(std::abs(projMax - projMin));

    // Étape 4 : décalage du centre vers le début du volume
    Z3i::Point center = moveCenterAlongDirection(ptC, -normalDir, maxScan / 2);

    // Étape 5 : initialisation de l'image résultat
    Image2D result(aDomain2D);
    for (auto it = result.domain().begin(); it != result.domain().end(); ++it)
        result.setValue(*it, 0);

    // Étape 6 : embedder et extraction
    DGtal::functors::Point2DEmbedderIn3D<Z3i::Domain> embedder(
        meshVolImage.domain(), center, normalDir, widthImageScan);

    DGtal::functors::Identity idV;
    ImageAdapterExtractor extractedImage(meshVolImage, aDomain2D, embedder, idV);

    int k = 0;
    bool firstFound = false;

    // Étape 7 : balayage progressif
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