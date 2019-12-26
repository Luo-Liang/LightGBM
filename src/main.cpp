/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/application.h>
#include <easy/profiler.h>
#include <iostream>
#include <thread>
#include <chrono>

int main(int argc, char **argv)
{
  if (NULL != std::getenv("LIGHTGBM_ENABLE_PROFILER"))
  {
    EASY_PROFILER_ENABLE;
  }
  else
  {
    EASY_PROFILER_DISABLE;
  }
  try
  {
    auto launchWait = std::getenv("LAUNCH_WAIT");
    if(launchWait != NULL)
    {
      ///activate launch wait 
      std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    }
    EASY_PROFILER_ENABLE;
    LightGBM::Application app(argc, argv);
    app.Run();
    auto envFileName = std::getenv("LIGHTGBM_PROFILE_NAME");
    std::string fName(envFileName == NULL ? "" : envFileName);
    if (fName == "")
    {
      char filename[2048];
      tm *timenow;
      time_t now = time(NULL);
      timenow = gmtime(&now);
      strftime(filename, sizeof(filename), "lightgbm_%Y-%m-%d_%H-%M-%S.prof", timenow);
      fName = std::string(filename);
    }
    profiler::dumpBlocksToFile(fName.c_str());
  }
  catch (const std::exception &ex)
  {
    std::cerr << "Met Exceptions:" << std::endl;
    std::cerr << ex.what() << std::endl;
    exit(-1);
  }
  catch (const std::string &ex)
  {
    std::cerr << "Met Exceptions:" << std::endl;
    std::cerr << ex << std::endl;
    exit(-1);
  }
  catch (...)
  {
    std::cerr << "Unknown Exceptions" << std::endl;
    exit(-1);
  }
}
